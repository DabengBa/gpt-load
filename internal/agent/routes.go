package agent

import (
	"net/url"
	"sort"

	"github.com/gin-gonic/gin"

	"gpt-load/internal/execution"
	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/platform/response"
	"gpt-load/internal/protocol"
	"gpt-load/internal/state"
)

func (server *Server) handleListRoutes(c *gin.Context) {
	if !requireScope(c, ScopeDiagnosticsRead) {
		return
	}
	snapshot := server.currentSnapshot()
	if snapshot == nil {
		writeAgentServiceError(c, "agent_list_routes", app_errors.ErrInternalServer)
		return
	}
	values, err := url.ParseQuery(c.Request.URL.RawQuery)
	if err != nil {
		writeAgentServiceError(c, "agent_list_routes", app_errors.ErrBadRequest)
		return
	}
	allowed := map[string]struct{}{"external_model": {}, "protocol": {}, "operation": {}}
	for key, value := range values {
		if _, ok := allowed[key]; !ok || len(value) != 1 {
			writeAgentServiceError(c, "agent_list_routes", app_errors.ErrBadRequest)
			return
		}
	}
	externalModel, hasExternalModel := agentQueryValue(values, "external_model")
	if !hasExternalModel || externalModel == "" {
		if _, protocolSet := agentQueryValue(values, "protocol"); protocolSet {
			writeAgentServiceError(c, "agent_list_routes", app_errors.ErrValidation)
			return
		}
		if _, operationSet := agentQueryValue(values, "operation"); operationSet {
			writeAgentServiceError(c, "agent_list_routes", app_errors.ErrValidation)
			return
		}
		response.SuccessI18n(c, "common.success", buildRouteIndex(snapshot))
		return
	}
	if !validAgentModel(externalModel) || externalModel == state.NoModelRouteKey {
		writeAgentServiceError(c, "agent_list_routes", app_errors.ErrValidation)
		return
	}
	protocolValue, hasProtocol := agentQueryValue(values, "protocol")
	operationValue, hasOperation := agentQueryValue(values, "operation")
	if !hasProtocol || !hasOperation {
		writeAgentServiceError(c, "agent_list_routes", app_errors.ErrValidation)
		return
	}
	protocolKey := protocol.Protocol(protocolValue)
	operationKey := execution.Operation(operationValue)
	if !protocolKey.Valid() || !operationKey.Valid() {
		writeAgentServiceError(c, "agent_list_routes", app_errors.ErrValidation)
		return
	}
	targets, exists := snapshot.ExecutionRouteCatalog[protocolKey][operationKey][externalModel]
	if !exists || len(targets) == 0 {
		writeAgentServiceError(c, "agent_list_routes", app_errors.ErrResourceNotFound)
		return
	}
	response.SuccessI18n(c, "common.success", buildRouteDetail(
		snapshot,
		protocolKey,
		operationKey,
		externalModel,
		targets,
	))
}

func (server *Server) currentSnapshot() *state.ConfigSnapshot {
	if server == nil || server.snapshots == nil {
		return nil
	}
	return server.snapshots.Current()
}

func buildRouteIndex(snapshot *state.ConfigSnapshot) RoutesView {
	type indexKey struct {
		external  string
		protocol  protocol.Protocol
		operation execution.Operation
	}
	type accumulator struct {
		candidates map[state.RouteEntryKey]struct{}
		groups     map[uint]struct{}
	}
	accumulators := make(map[indexKey]*accumulator)
	for protocolKey, byOperation := range snapshot.ExecutionRouteCatalog {
		for operationKey, byModel := range byOperation {
			for external, targets := range byModel {
				if external == state.NoModelRouteKey || len(targets) == 0 {
					continue
				}
				key := indexKey{external: external, protocol: protocolKey, operation: operationKey}
				entry, exists := accumulators[key]
				if !exists {
					entry = &accumulator{
						candidates: make(map[state.RouteEntryKey]struct{}),
						groups:     make(map[uint]struct{}),
					}
					accumulators[key] = entry
				}
				for _, target := range targets {
					candidateKey := state.RouteEntryKey{GroupID: target.GroupID, EntryID: target.EntryID}
					if _, duplicate := entry.candidates[candidateKey]; duplicate {
						continue
					}
					entry.candidates[candidateKey] = struct{}{}
					entry.groups[target.GroupID] = struct{}{}
				}
			}
		}
	}
	items := make([]RouteIndexItemView, 0, len(accumulators))
	for key, entry := range accumulators {
		groupIDs := make([]uint, 0, len(entry.groups))
		for groupID := range entry.groups {
			groupIDs = append(groupIDs, groupID)
		}
		sort.Slice(groupIDs, func(i, j int) bool { return groupIDs[i] < groupIDs[j] })
		items = append(items, RouteIndexItemView{
			ExternalModel:  key.external,
			Protocol:       string(key.protocol),
			Operation:      string(key.operation),
			CandidateCount: len(entry.candidates),
			GroupIDs:       groupIDs,
		})
	}
	sort.Slice(items, func(i, j int) bool {
		if items[i].ExternalModel != items[j].ExternalModel {
			return items[i].ExternalModel < items[j].ExternalModel
		}
		if items[i].Protocol != items[j].Protocol {
			return items[i].Protocol < items[j].Protocol
		}
		return items[i].Operation < items[j].Operation
	})
	return RoutesView{
		SchemaVersion:    SchemaVersion,
		View:             RoutesViewIndex,
		SnapshotRevision: snapshot.Revision,
		Items:            items,
	}
}

func buildRouteDetail(
	snapshot *state.ConfigSnapshot,
	protocolKey protocol.Protocol,
	operationKey execution.Operation,
	externalModel string,
	targets []state.RouteTarget,
) RoutesView {
	seen := make(map[state.RouteEntryKey]struct{}, len(targets))
	entries := make([]RouteEntryView, 0, len(targets))
	for _, target := range targets {
		key := state.RouteEntryKey{GroupID: target.GroupID, EntryID: target.EntryID}
		if _, duplicate := seen[key]; duplicate {
			continue
		}
		seen[key] = struct{}{}
		entries = append(entries, RouteEntryView{
			GroupID:         target.GroupID,
			GroupName:       groupName(snapshot, target.GroupID),
			EntryID:         target.EntryID,
			UpstreamModelID: target.UpstreamModelID,
			RouteMode:       string(target.Mode),
			Weight:          target.EntryWeight,
			Priority:        target.Priority,
		})
	}
	sort.Slice(entries, func(i, j int) bool {
		if entries[i].Priority != entries[j].Priority {
			return entries[i].Priority < entries[j].Priority
		}
		if entries[i].GroupID != entries[j].GroupID {
			return entries[i].GroupID < entries[j].GroupID
		}
		return entries[i].EntryID < entries[j].EntryID
	})
	detail := &RouteDetailView{
		ExternalModel: externalModel,
		Protocol:      string(protocolKey),
		Operation:     string(operationKey),
		Entries:       entries,
	}
	return RoutesView{
		SchemaVersion:    SchemaVersion,
		View:             RoutesViewDetail,
		SnapshotRevision: snapshot.Revision,
		Detail:           detail,
	}
}

func groupName(snapshot *state.ConfigSnapshot, groupID uint) string {
	if snapshot == nil {
		return ""
	}
	if catalog, exists := snapshot.GroupCatalog[groupID]; exists {
		return catalog.Name
	}
	if group, exists := snapshot.Groups[groupID]; exists {
		return group.Name
	}
	if group, exists := snapshot.DisabledGroups[groupID]; exists {
		return group.Name
	}
	return ""
}
