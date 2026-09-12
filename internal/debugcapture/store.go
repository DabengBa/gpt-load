//go:build aix || darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris

package debugcapture

import (
	"archive/zip"
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"path/filepath"
	"time"

	"gorm.io/gorm"
	"gorm.io/gorm/clause"

	"gpt-load/internal/storage/dbtx"
	"gpt-load/internal/storage/models"
)

const (
	retention        = 12 * time.Hour
	bodyChunkSize    = 64 * 1024
	cleanupBatchSize = 100
)

type Store struct {
	db  *gorm.DB
	now func() time.Time
}

type Session struct {
	store *Store
	id    string
}

type Attempt struct {
	session *Session
	id      string
}

func New(db *gorm.DB) (*Store, error) {
	return NewWithClock(db, time.Now)
}

func NewWithClock(db *gorm.DB, now func() time.Time) (*Store, error) {
	if db == nil {
		return nil, errors.New("debug capture database is required")
	}
	if now == nil {
		return nil, errors.New("debug capture clock is required")
	}
	return &Store{db: db, now: now}, nil
}

func (s *Store) StartSession(metadata SessionMetadata) (*Session, error) {
	if s == nil || s.db == nil {
		return nil, errors.New("debug capture database is required")
	}
	now := s.now().UTC()
	id, err := newID()
	if err != nil {
		return nil, fmt.Errorf("create debug capture session id: %w", err)
	}
	rawMetadata, err := encodeMetadata(metadata)
	if err != nil {
		return nil, err
	}
	row := models.DebugCapture{
		ID: id, RequestID: metadata.RequestID, AccessKeyID: metadata.AccessKeyID,
		Protocol: metadata.Protocol, Operation: metadata.Operation, Metadata: rawMetadata,
		CreatedAtMS: now.UnixMilli(), ExpiresAtMS: now.Add(retention).UnixMilli(), State: string(StateActive),
	}
	if err := s.write("start debug capture session", func(tx *gorm.DB) error {
		return tx.Create(&row).Error
	}); err != nil {
		return nil, fmt.Errorf("create debug capture session: %w", err)
	}
	return &Session{store: s, id: id}, nil
}

// ListSessions keeps the original list surface while accepting an optional
// filter for callers that need the paged admin query contract.
func (s *Store) ListSessions(filters ...SessionQuery) ([]SessionRecord, error) {
	filter := SessionQuery{Limit: -1}
	if len(filters) > 0 {
		filter = filters[0]
	}
	return s.QuerySessions(filter)
}

func (s *Store) QuerySessions(filter SessionQuery) ([]SessionRecord, error) {
	var rows []models.DebugCapture
	err := s.read("query debug capture sessions", func(tx *gorm.DB) error {
		query := tx.Model(&models.DebugCapture{}).Where("expires_at_ms > ?", s.now().UTC().UnixMilli()).Order("created_at_ms ASC, id ASC")
		if filter.RequestID != "" {
			query = query.Where("request_id = ?", filter.RequestID)
		}
		if filter.AccessKeyID != 0 {
			query = query.Where("access_key_id = ?", filter.AccessKeyID)
		}
		if filter.Protocol != "" {
			query = query.Where("protocol = ?", filter.Protocol)
		}
		if filter.Operation != "" {
			query = query.Where("operation = ?", filter.Operation)
		}
		if filter.State != "" {
			query = query.Where("state = ?", filter.State)
		}
		if !filter.CreatedAfter.IsZero() {
			query = query.Where("created_at_ms >= ?", filter.CreatedAfter.UTC().UnixMilli())
		}
		if !filter.CreatedBefore.IsZero() {
			query = query.Where("created_at_ms < ?", filter.CreatedBefore.UTC().UnixMilli())
		}
		if filter.Offset > 0 {
			query = query.Offset(filter.Offset)
		}
		if filter.Limit >= 0 {
			query = query.Limit(filter.Limit)
		}
		return query.Preload("Attempts", func(db *gorm.DB) *gorm.DB {
			return db.Order("sequence ASC")
		}).Find(&rows).Error
	})
	if err != nil {
		return nil, fmt.Errorf("query debug capture sessions: %w", err)
	}
	result := make([]SessionRecord, 0, len(rows))
	for _, row := range rows {
		record, err := sessionRecord(row)
		if err != nil {
			return nil, err
		}
		result = append(result, record)
	}
	return result, nil
}

func (s *Store) ReadSession(id string) (SessionRecord, error) {
	var row models.DebugCapture
	err := s.read("read debug capture session", func(tx *gorm.DB) error {
		return tx.Preload("Attempts", func(db *gorm.DB) *gorm.DB {
			return db.Order("sequence ASC")
		}).First(&row, "id = ?", id).Error
	})
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return SessionRecord{}, ErrNotFound
	}
	if err != nil {
		return SessionRecord{}, fmt.Errorf("read debug capture session: %w", err)
	}
	if s.expired(row, s.now()) {
		return SessionRecord{}, s.markExpiryOnError(id, ErrExpired)
	}
	return sessionRecord(row)
}

func (s *Store) Counts() (Counts, error) {
	if s == nil || s.db == nil {
		return Counts{}, errors.New("debug capture database is required")
	}
	var rows []struct {
		State string
		Count int64
	}
	err := s.read("count debug captures", func(tx *gorm.DB) error {
		return tx.Model(&models.DebugCapture{}).
			Select("state, COUNT(*) AS count").
			Where("expires_at_ms > ?", s.now().UTC().UnixMilli()).
			Group("state").
			Scan(&rows).Error
	})
	if err != nil {
		return Counts{}, fmt.Errorf("count debug captures: %w", err)
	}
	result := Counts{}
	for _, row := range rows {
		if row.Count < 0 {
			return Counts{}, errors.New("count debug captures: negative count")
		}
		switch State(row.State) {
		case StateActive:
			result.Active = row.Count
		case StateCompleted:
			result.Completed = row.Count
		case StateFailed:
			result.Failed = row.Count
		}
	}
	return result, nil
}

func (s *Store) Cleanup() (int, error) {
	removed := 0
	nowMS := s.now().UTC().UnixMilli()
	for {
		var batch []string
		err := s.write("clean up debug captures", func(tx *gorm.DB) error {
			return tx.Model(&models.DebugCapture{}).Where("expires_at_ms <= ?", nowMS).
				Order("expires_at_ms ASC, id ASC").Limit(cleanupBatchSize).Pluck("id", &batch).Error
		})
		if err != nil {
			return removed, fmt.Errorf("list expired debug captures: %w", err)
		}
		if len(batch) == 0 {
			return removed, nil
		}
		var deleted int64
		if err := s.write("delete expired debug captures", func(tx *gorm.DB) error {
			result := tx.Where("id IN ?", batch).Delete(&models.DebugCapture{})
			deleted = result.RowsAffected
			return result.Error
		}); err != nil {
			return removed, fmt.Errorf("delete expired debug captures: %w", err)
		}
		removed += int(deleted)
	}
}

func (s *Store) ExportZIP(id string, output io.Writer) error {
	if output == nil {
		return errors.New("debug capture export writer is required")
	}
	err := s.read("export debug capture", func(tx *gorm.DB) error {
		var row models.DebugCapture
		if err := tx.Preload("Attempts", func(db *gorm.DB) *gorm.DB {
			return db.Order("sequence ASC")
		}).First(&row, "id = ?", id).Error; err != nil {
			if errors.Is(err, gorm.ErrRecordNotFound) {
				return ErrNotFound
			}
			return err
		}
		if s.expired(row, s.now()) {
			return ErrExpired
		}
		record, err := sessionRecord(row)
		if err != nil {
			return err
		}
		archive := zip.NewWriter(output)
		if err := addJSONToZIP(archive, "manifest.json", record); err != nil {
			_ = archive.Close()
			return fmt.Errorf("write debug capture manifest: %w", err)
		}
		for _, attempt := range row.Attempts {
			metadata, err := attemptRecord(attempt)
			if err != nil {
				_ = archive.Close()
				return err
			}
			if err := addJSONToZIP(archive, filepath.Join("attempts", attempt.ID, "metadata.json"), metadata); err != nil {
				_ = archive.Close()
				return err
			}
			for _, direction := range []Direction{DirectionRequest, DirectionResponse} {
				for _, part := range []Part{PartHeaders, PartBody} {
					if err := addChunksToZIP(tx, archive, row.ID, attempt.ID, part, direction); err != nil {
						_ = archive.Close()
						return err
					}
				}
			}
		}
		if err := archive.Close(); err != nil {
			return fmt.Errorf("close debug capture export: %w", err)
		}
		return nil
	})
	return s.markExpiryOnError(id, err)
}

func (s *Store) ReadPart(sessionID, attemptID string, part Part, direction Direction) ([]byte, error) {
	if err := validPart(part, direction); err != nil {
		return nil, err
	}
	var content bytes.Buffer
	found := false
	err := s.read("read debug capture part", func(tx *gorm.DB) error {
		var session models.DebugCapture
		if err := tx.First(&session, "id = ?", sessionID).Error; err != nil {
			if errors.Is(err, gorm.ErrRecordNotFound) {
				return ErrNotFound
			}
			return err
		}
		if s.expired(session, s.now()) {
			return ErrExpired
		}
		if err := streamChunks(tx, sessionID, attemptID, part, direction, &content, &found); err != nil {
			return err
		}
		if !found {
			return ErrNotFound
		}
		return nil
	})
	if errors.Is(err, ErrExpired) {
		err = s.markExpiryOnError(sessionID, err)
	}
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, ErrNotFound
	}
	if err != nil {
		return nil, err
	}
	return content.Bytes(), nil
}

func (s *Store) startAttempt(sessionID string, metadata AttemptMetadata) (*Attempt, error) {
	rawMetadata, err := encodeMetadata(metadata)
	if err != nil {
		return nil, err
	}
	id, err := newID()
	if err != nil {
		return nil, fmt.Errorf("create debug capture attempt id: %w", err)
	}
	var created models.DebugCaptureAttempt
	err = s.write("start debug capture attempt", func(tx *gorm.DB) error {
		var session models.DebugCapture
		if err := lockSession(tx, sessionID, &session); err != nil {
			return classifyLookupError(err)
		}
		now := s.now().UTC()
		if err := s.requireSessionActive(tx, &session, now); err != nil {
			return err
		}
		var sequence int
		if err := tx.Model(&models.DebugCaptureAttempt{}).Where("capture_id = ?", sessionID).Select("COALESCE(MAX(sequence), 0)").Scan(&sequence).Error; err != nil {
			return err
		}
		created = models.DebugCaptureAttempt{ID: id, CaptureID: sessionID, Sequence: sequence + 1, Metadata: rawMetadata, StartedAtMS: now.UnixMilli(), State: string(StateActive)}
		return tx.Create(&created).Error
	})
	if err != nil {
		return nil, s.markExpiryOnError(sessionID, err)
	}
	return &Attempt{session: &Session{store: s, id: sessionID}, id: id}, nil
}

func (s *Store) appendPart(sessionID, attemptID string, part Part, direction Direction, reader io.Reader) error {
	if reader == nil {
		return errors.New("debug capture part reader is required")
	}
	if err := validPart(part, direction); err != nil {
		return err
	}
	err := s.write("append debug capture part", func(tx *gorm.DB) error {
		var session models.DebugCapture
		if err := lockSession(tx, sessionID, &session); err != nil {
			return classifyLookupError(err)
		}
		now := s.now().UTC()
		if err := s.requireSessionActive(tx, &session, now); err != nil {
			return err
		}
		var attempt models.DebugCaptureAttempt
		if err := lockAttempt(tx, sessionID, attemptID, &attempt); err != nil {
			return classifyLookupError(err)
		}
		if attempt.State != string(StateActive) {
			return ErrClosed
		}
		buffer := make([]byte, bodyChunkSize)
		wrote := false
		for {
			n, readErr := reader.Read(buffer)
			if n > 0 {
				now = s.now().UTC()
				if err := s.requireSessionActive(tx, &session, now); err != nil {
					return err
				}
				data := append([]byte(nil), buffer[:n]...)
				if err := tx.Create(&models.DebugCaptureChunk{
					CaptureID: sessionID, AttemptID: attemptID, Part: string(part), Direction: string(direction), Data: data, CreatedAtMS: now.UnixMilli(),
				}).Error; err != nil {
					return err
				}
				wrote = true
			}
			if errors.Is(readErr, io.EOF) {
				break
			}
			if readErr != nil {
				return readErr
			}
		}
		now = s.now().UTC()
		if err := s.requireSessionActive(tx, &session, now); err != nil {
			return err
		}
		if !wrote && part == PartHeaders {
			return tx.Create(&models.DebugCaptureChunk{CaptureID: sessionID, AttemptID: attemptID, Part: string(part), Direction: string(direction), Data: []byte{}, CreatedAtMS: now.UnixMilli()}).Error
		}
		return nil
	})
	return s.markExpiryOnError(sessionID, err)
}

func (s *Store) updateSessionMetadata(id string, patch SessionMetadata) error {
	err := s.write("update debug capture session metadata", func(tx *gorm.DB) error {
		var session models.DebugCapture
		if err := lockSession(tx, id, &session); err != nil {
			return classifyLookupError(err)
		}
		now := s.now().UTC()
		if err := s.requireSessionActive(tx, &session, now); err != nil {
			return err
		}
		metadata, err := decodeSessionMetadata(session.Metadata)
		if err != nil {
			return err
		}
		if patch.RequestID != "" {
			session.RequestID = patch.RequestID
			metadata.RequestID = patch.RequestID
		}
		if patch.AccessKeyID != 0 {
			session.AccessKeyID = patch.AccessKeyID
			metadata.AccessKeyID = patch.AccessKeyID
		}
		if patch.Protocol != "" {
			session.Protocol = patch.Protocol
			metadata.Protocol = patch.Protocol
		}
		if patch.Operation != "" {
			session.Operation = patch.Operation
			metadata.Operation = patch.Operation
		}
		if len(patch.Fields) > 0 {
			if metadata.Fields == nil {
				metadata.Fields = make(map[string]any, len(patch.Fields))
			}
			for key, value := range patch.Fields {
				metadata.Fields[key] = value
			}
		}
		rawMetadata, err := encodeMetadata(metadata)
		if err != nil {
			return err
		}
		return tx.Model(&models.DebugCapture{}).Where("id = ? AND state = ?", id, StateActive).
			Updates(map[string]any{
				"request_id":    patchValue(patch.RequestID, session.RequestID),
				"access_key_id": session.AccessKeyID,
				"protocol":      session.Protocol,
				"operation":     session.Operation,
				"metadata":      rawMetadata,
			}).Error
	})
	return s.markExpiryOnError(id, err)
}

func patchValue(patch, current string) string {
	if patch != "" {
		return patch
	}
	return current
}
func (s *Store) updateSession(id string, state State, message string) error {
	err := s.write("finish debug capture session", func(tx *gorm.DB) error {
		var session models.DebugCapture
		if err := lockSession(tx, id, &session); err != nil {
			return classifyLookupError(err)
		}
		now := s.now().UTC()
		if err := s.requireSessionActive(tx, &session, now); err != nil {
			return err
		}
		terminalAt := now.UnixMilli()
		result := tx.Model(&models.DebugCapture{}).Where("id = ? AND state = ? AND expires_at_ms > ?", id, StateActive, terminalAt).
			Updates(map[string]any{"state": state, "error": message, "terminal_at_ms": terminalAt})
		if result.Error != nil {
			return result.Error
		}
		if result.RowsAffected != 1 {
			return s.classifySession(tx, id, now)
		}
		return nil
	})
	return s.markExpiryOnError(id, err)
}

func (s *Store) updateAttempt(sessionID, attemptID string, state State, message string) error {
	err := s.write("finish debug capture attempt", func(tx *gorm.DB) error {
		var session models.DebugCapture
		if err := lockSession(tx, sessionID, &session); err != nil {
			return classifyLookupError(err)
		}
		now := s.now().UTC()
		if err := s.requireSessionActive(tx, &session, now); err != nil {
			return err
		}
		var attempt models.DebugCaptureAttempt
		if err := lockAttempt(tx, sessionID, attemptID, &attempt); err != nil {
			return classifyLookupError(err)
		}
		if attempt.State != string(StateActive) {
			return ErrClosed
		}
		terminalAt := now.UnixMilli()
		result := tx.Model(&models.DebugCaptureAttempt{}).Where("id = ? AND capture_id = ? AND state = ?", attemptID, sessionID, StateActive).
			Updates(map[string]any{"state": state, "error": message, "completed_at_ms": terminalAt})
		if result.Error != nil {
			return result.Error
		}
		if result.RowsAffected != 1 {
			return ErrClosed
		}
		return nil
	})
	return s.markExpiryOnError(sessionID, err)
}

func (s *Store) requireSessionActive(tx *gorm.DB, session *models.DebugCapture, now time.Time) error {
	if s.expired(*session, now) {
		return ErrExpired
	}
	if session.State != string(StateActive) {
		return ErrClosed
	}
	return nil
}

func (s *Store) markExpiryOnError(id string, err error) error {
	if !errors.Is(err, ErrExpired) {
		return err
	}
	if markErr := s.markExpired(id); markErr != nil {
		return fmt.Errorf("%w: persist expired state: %v", ErrExpired, markErr)
	}
	return err
}

func (s *Store) classifySession(tx *gorm.DB, id string, now time.Time) error {
	var session models.DebugCapture
	if err := tx.First(&session, "id = ?", id).Error; err != nil {
		return classifyLookupError(err)
	}
	return s.requireSessionActive(tx, &session, now)
}

func (s *Store) markExpired(id string) error {
	return s.write("mark expired debug capture", func(tx *gorm.DB) error {
		result := tx.Model(&models.DebugCapture{}).Where("id = ? AND state = ? AND expires_at_ms <= ?", id, StateActive, s.now().UTC().UnixMilli()).Update("state", StateExpired)
		if result.Error != nil {
			return result.Error
		}
		if result.RowsAffected == 1 {
			return nil
		}
		var row models.DebugCapture
		if err := tx.First(&row, "id = ?", id).Error; err != nil {
			return classifyLookupError(err)
		}
		if row.State == string(StateExpired) {
			return nil
		}
		return fmt.Errorf("debug capture %q was not marked expired", id)
	})
}

func (s *Store) write(operation string, callback func(*gorm.DB) error) error {
	return dbtx.Run(nil, s.db, dbtx.Options{Mode: dbtx.Write, Operation: operation}, callback)
}

func (s *Store) read(operation string, callback func(*gorm.DB) error) error {
	return dbtx.Run(nil, s.db, dbtx.Options{Mode: dbtx.ReadSnapshot, Operation: operation}, callback)
}

func lockSession(tx *gorm.DB, id string, session *models.DebugCapture) error {
	return tx.Clauses(clause.Locking{Strength: "UPDATE"}).First(session, "id = ?", id).Error
}

func lockAttempt(tx *gorm.DB, sessionID, attemptID string, attempt *models.DebugCaptureAttempt) error {
	return tx.Clauses(clause.Locking{Strength: "UPDATE"}).Where("capture_id = ?", sessionID).First(attempt, "id = ?", attemptID).Error
}

func classifyLookupError(err error) error {
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return ErrNotFound
	}
	return err
}

func (s *Store) expired(row models.DebugCapture, now time.Time) bool {
	return !now.UTC().Before(time.UnixMilli(row.ExpiresAtMS))
}

func sessionRecord(row models.DebugCapture) (SessionRecord, error) {
	metadata, err := decodeSessionMetadata(row.Metadata)
	if err != nil {
		return SessionRecord{}, err
	}
	record := SessionRecord{ID: row.ID, RequestID: row.RequestID, AccessKeyID: row.AccessKeyID, Protocol: row.Protocol, Operation: row.Operation, Metadata: metadata, CreatedAt: time.UnixMilli(row.CreatedAtMS).UTC(), ExpiresAt: time.UnixMilli(row.ExpiresAtMS).UTC(), State: State(row.State), Error: row.Error, Attempts: make([]AttemptRecord, 0, len(row.Attempts))}
	for _, attempt := range row.Attempts {
		converted, err := attemptRecord(attempt)
		if err != nil {
			return SessionRecord{}, err
		}
		record.Attempts = append(record.Attempts, converted)
	}
	return record, nil
}

func attemptRecord(row models.DebugCaptureAttempt) (AttemptRecord, error) {
	metadata, err := decodeAttemptMetadata(row.Metadata)
	if err != nil {
		return AttemptRecord{}, err
	}
	record := AttemptRecord{ID: row.ID, Sequence: row.Sequence, Metadata: metadata, StartedAt: time.UnixMilli(row.StartedAtMS).UTC(), State: State(row.State), Error: row.Error}
	if row.CompletedAtMS != nil {
		record.CompletedAt = time.UnixMilli(*row.CompletedAtMS).UTC()
	}
	return record, nil
}

func encodeMetadata(value any) (models.JSON, error) {
	content, err := json.Marshal(value)
	if err != nil {
		return nil, fmt.Errorf("encode debug capture metadata: %w", err)
	}
	return models.JSON(content), nil
}

func decodeSessionMetadata(raw models.JSON) (SessionMetadata, error) {
	var result SessionMetadata
	if len(raw) == 0 {
		return result, nil
	}
	if err := json.Unmarshal(raw, &result); err != nil {
		return result, fmt.Errorf("decode debug capture session metadata: %w", err)
	}
	return result, nil
}

func decodeAttemptMetadata(raw models.JSON) (AttemptMetadata, error) {
	var result AttemptMetadata
	if len(raw) == 0 {
		return result, nil
	}
	if err := json.Unmarshal(raw, &result); err != nil {
		return result, fmt.Errorf("decode debug capture attempt metadata: %w", err)
	}
	return result, nil
}

func streamChunks(tx *gorm.DB, captureID, attemptID string, part Part, direction Direction, output io.Writer, found *bool) error {
	rows, err := tx.Model(&models.DebugCaptureChunk{}).Select("data").Where("capture_id = ? AND attempt_id = ? AND part = ? AND direction = ?", captureID, attemptID, part, direction).Order("id ASC").Rows()
	if err != nil {
		return err
	}
	defer rows.Close()
	for rows.Next() {
		*found = true
		var data []byte
		if err := rows.Scan(&data); err != nil {
			return err
		}
		if _, err := output.Write(data); err != nil {
			return err
		}
	}
	return rows.Err()
}

func addChunksToZIP(tx *gorm.DB, archive *zip.Writer, captureID, attemptID string, part Part, direction Direction) error {
	header := &zip.FileHeader{Name: archivePartName(attemptID, part, direction), Method: zip.Deflate}
	header.SetMode(0600)
	writer, err := archive.CreateHeader(header)
	if err != nil {
		return err
	}
	found := false
	if err := streamChunks(tx, captureID, attemptID, part, direction, writer, &found); err != nil {
		return fmt.Errorf("stream debug capture part: %w", err)
	}
	return nil
}

func (s *Session) ID() string { return s.id }
func (s *Session) StartAttempt(metadata AttemptMetadata) (*Attempt, error) {
	return s.store.startAttempt(s.id, metadata)
}
func (s *Session) UpdateMetadata(patch SessionMetadata) error {
	if s == nil || s.store == nil {
		return errors.New("debug capture session is required")
	}
	return s.store.updateSessionMetadata(s.id, patch)
}
func (s *Session) Complete() error { return s.store.updateSession(s.id, StateCompleted, "") }
func (s *Session) Fail(err error) error {
	message := "capture failed"
	if err != nil {
		message = err.Error()
	}
	return s.store.updateSession(s.id, StateFailed, message)
}

func (s *Store) appendAttemptEvent(sessionID, attemptID string, event EventRecord) error {
	err := s.write("append debug capture attempt event", func(tx *gorm.DB) error {
		var session models.DebugCapture
		if err := lockSession(tx, sessionID, &session); err != nil {
			return classifyLookupError(err)
		}
		now := s.now().UTC()
		if err := s.requireSessionActive(tx, &session, now); err != nil {
			return err
		}
		var attempt models.DebugCaptureAttempt
		if err := lockAttempt(tx, sessionID, attemptID, &attempt); err != nil {
			return classifyLookupError(err)
		}
		if attempt.State != string(StateActive) {
			return ErrClosed
		}
		metadata, err := decodeAttemptMetadata(attempt.Metadata)
		if err != nil {
			return err
		}
		metadata.Events = append(metadata.Events, event)
		rawMetadata, err := encodeMetadata(metadata)
		if err != nil {
			return err
		}
		result := tx.Model(&models.DebugCaptureAttempt{}).
			Where("id = ? AND capture_id = ? AND state = ?", attemptID, sessionID, StateActive).
			Update("metadata", rawMetadata)
		if result.Error != nil {
			return result.Error
		}
		if result.RowsAffected != 1 {
			return ErrClosed
		}
		return nil
	})
	return s.markExpiryOnError(sessionID, err)
}

func (a *Attempt) ID() string { return a.id }
func (a *Attempt) AppendHeaders(direction Direction, data []byte) error {
	return a.session.store.appendPart(a.session.id, a.id, PartHeaders, direction, bytes.NewReader(data))
}
func (a *Attempt) AppendBodyPart(direction Direction, reader io.Reader) error {
	return a.session.store.appendPart(a.session.id, a.id, PartBody, direction, reader)
}

func (a *Attempt) RecordResponseFlush() error {
	return a.recordEvent(EventRecord{Kind: "response_flush"})
}

func (a *Attempt) RecordResponseShortWrite(written, requested int) error {
	return a.recordEvent(EventRecord{
		Kind: "response_short_write", Written: written, Requested: requested,
	})
}

func (a *Attempt) RecordResponseError(err error) error {
	return a.recordEvent(EventRecord{Kind: "response_error", Error: errorText(err)})
}

func (a *Attempt) RecordResponseTermination(termination, detail string) error {
	return a.recordEvent(EventRecord{Kind: "response_termination", Outcome: termination, Error: detail})
}

func (a *Attempt) RecordResponseHijack(err error) error {
	return a.recordEvent(EventRecord{Kind: "response_hijack", Error: errorText(err)})
}

func (a *Attempt) RecordRequestError(err error) error {
	return a.recordEvent(EventRecord{Kind: "request_error", Error: errorText(err)})
}

func (a *Attempt) RecordRequestClose(err error) error {
	return a.recordEvent(EventRecord{Kind: "request_close", Error: errorText(err)})
}

func (a *Attempt) RecordRequestOutcome(kind string, err error) error {
	return a.recordEvent(EventRecord{
		Kind: "request_outcome", Outcome: kind, Error: errorText(err),
	})
}

func (a *Attempt) recordEvent(event EventRecord) error {
	if a == nil || a.session == nil || a.session.store == nil {
		return errors.New("debug capture attempt is required")
	}
	event.At = a.session.store.now().UTC()
	return a.session.store.appendAttemptEvent(a.session.id, a.id, event)
}

func errorText(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

func (a *Attempt) Complete() error {
	return a.session.store.updateAttempt(a.session.id, a.id, StateCompleted, "")
}
func (a *Attempt) Fail(err error) error {
	message := "capture failed"
	if err != nil {
		message = err.Error()
	}
	return a.session.store.updateAttempt(a.session.id, a.id, StateFailed, message)
}

func validPart(part Part, direction Direction) error {
	if part != PartHeaders && part != PartBody {
		return fmt.Errorf("invalid debug capture part %q", part)
	}
	if direction != DirectionRequest && direction != DirectionResponse {
		return fmt.Errorf("invalid debug capture direction %q", direction)
	}
	return nil
}

func archivePartName(attemptID string, part Part, direction Direction) string {
	return filepath.ToSlash(filepath.Join("attempts", attemptID, "parts", string(direction)+"."+string(part)))
}

func addJSONToZIP(archive *zip.Writer, name string, value any) error {
	content, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return err
	}
	header := &zip.FileHeader{Name: filepath.ToSlash(name), Method: zip.Deflate}
	header.SetMode(0600)
	writer, err := archive.CreateHeader(header)
	if err != nil {
		return err
	}
	_, err = writer.Write(content)
	return err
}

func newID() (string, error) {
	var data [16]byte
	if _, err := rand.Read(data[:]); err != nil {
		return "", err
	}
	return hex.EncodeToString(data[:]), nil
}
