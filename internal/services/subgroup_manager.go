package services

import (
	"fmt"
	"gpt-load/internal/models"
	"gpt-load/internal/store"
	"strings"
	"sync"

	"github.com/sirupsen/logrus"
)

// SubGroupManager manages weighted round-robin selection for all aggregate groups
type SubGroupManager struct {
	store     store.Store
	selectors map[uint]*selector
	mu        sync.RWMutex
}

// subGroupItem represents a sub-group with its weight and current weight for round-robin
type subGroupItem struct {
	name            string
	subGroupID      uint
	weight          int
	currentWeight   int
	supportedModels []string
}

// NewSubGroupManager creates a new sub-group manager service
func NewSubGroupManager(store store.Store) *SubGroupManager {
	return &SubGroupManager{
		store:     store,
		selectors: make(map[uint]*selector),
	}
}

// SelectSubGroup selects an appropriate sub-group for the given aggregate group
func (m *SubGroupManager) SelectSubGroup(group *models.Group, modelName string) (string, error) {
	if group.GroupType != "aggregate" {
		return "", nil
	}

	selector := m.getSelector(group)
	if selector == nil {
		return "", fmt.Errorf("no valid sub-groups available for aggregate group '%s'", group.Name)
	}

	selectedName := selector.selectNext(modelName)
	if selectedName == "" {
		return "", fmt.Errorf("no suitable sub-groups found for aggregate group '%s' (model: %s)", group.Name, modelName)
	}

	logrus.WithFields(logrus.Fields{
		"aggregate_group": group.Name,
		"selected_group":  selectedName,
		"model":           modelName,
	}).Debug("Selected sub-group from aggregate")

	return selectedName, nil
}

// RebuildSelectors rebuild all selectors based on the incoming group
func (m *SubGroupManager) RebuildSelectors(groups map[string]*models.Group) {
	newSelectors := make(map[uint]*selector)

	for _, group := range groups {
		if group.GroupType == "aggregate" && len(group.SubGroups) > 0 {
			if sel := m.createSelector(group); sel != nil {
				newSelectors[group.ID] = sel
			}
		}
	}

	m.mu.Lock()
	m.selectors = newSelectors
	m.mu.Unlock()

	logrus.WithField("new_count", len(newSelectors)).Debug("Rebuilt selectors for aggregate groups")
}

// getSelector retrieves or creates a selector for the aggregate group
func (m *SubGroupManager) getSelector(group *models.Group) *selector {
	m.mu.RLock()
	if sel, exists := m.selectors[group.ID]; exists {
		m.mu.RUnlock()
		return sel
	}
	m.mu.RUnlock()

	m.mu.Lock()
	defer m.mu.Unlock()

	if sel, exists := m.selectors[group.ID]; exists {
		return sel
	}

	sel := m.createSelector(group)
	if sel != nil {
		m.selectors[group.ID] = sel
		logrus.WithFields(logrus.Fields{
			"group_id":        group.ID,
			"group_name":      group.Name,
			"sub_group_count": len(sel.subGroups),
		}).Debug("Created sub-group selector")
	}

	return sel
}

// createSelector creates a new selector for an aggregate group
func (m *SubGroupManager) createSelector(group *models.Group) *selector {
	if group.GroupType != "aggregate" || len(group.SubGroups) == 0 {
		return nil
	}

	var items []subGroupItem
	for _, sg := range group.SubGroups {
		items = append(items, subGroupItem{
			name:            sg.SubGroupName,
			subGroupID:      sg.SubGroupID,
			weight:          sg.Weight,
			currentWeight:   0,
			supportedModels: sg.SupportedModels,
		})
	}

	if len(items) == 0 {
		return nil
	}

	return &selector{
		groupID:   group.ID,
		groupName: group.Name,
		subGroups: items,
		store:     m.store,
	}
}

// selector encapsulates the weighted round-robin algorithm for a single aggregate group
type selector struct {
	groupID   uint
	groupName string
	subGroups []subGroupItem
	store     store.Store
	mu        sync.Mutex
}

// selectNext uses weighted round-robin algorithm to select a sub-group with active keys
func (s *selector) selectNext(modelName string) string {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.subGroups) == 0 {
		return ""
	}

	// Filter candidates by model support first
	candidatesIndices := make([]int, 0, len(s.subGroups))
	for i := range s.subGroups {
		if s.supportsModel(&s.subGroups[i], modelName) {
			candidatesIndices = append(candidatesIndices, i)
		}
	}

	if len(candidatesIndices) == 0 {
		logrus.WithFields(logrus.Fields{
			"aggregate_group": s.groupName,
			"model":           modelName,
		}).Warn("No sub-groups support the requested model")
		return ""
	}

	if len(candidatesIndices) == 1 {
		idx := candidatesIndices[0]
		if s.hasActiveKeys(s.subGroups[idx].subGroupID) {
			return s.subGroups[idx].name
		}
		return ""
	}

	attempted := make(map[uint]bool)
	// Only try candidates that support the model
	for len(attempted) < len(candidatesIndices) {
		// Use weighted selection only among filtered candidates
		// Note: The original selectByWeight implementation iterates ALL subGroups.
		// To correctly implement weighted selection on a SUBSET, we should probably rewrite selectByWeight
		// or just accept that we pick the "best" from the global list and verify if it's in our candidates.
		// However, doing that might skew weights if the "best" is constantly rejected because of model mismatch.
		// A better approach for subset weighted selection:
		// Calculate total weight of candidates, then pick.
		// But s.subGroups stores currentWeight state. Modifying that based on subset is tricky.
		// Simplified approach: Just loop through candidates using their current weights and pick the best one.

		item := s.selectByWeightFromSubset(candidatesIndices)
		if item == nil {
			break
		}

		if attempted[item.subGroupID] {
			// Should not happen with new logic, but safety first
			continue
		}
		attempted[item.subGroupID] = true

		if s.hasActiveKeys(item.subGroupID) {
			return item.name
		}
	}

	return ""
}

// selectByWeightFromSubset selects best item from specific indices
func (s *selector) selectByWeightFromSubset(indices []int) *subGroupItem {
	totalWeight := 0
	var best *subGroupItem

	for _, idx := range indices {
		item := &s.subGroups[idx]
		totalWeight += item.weight
		item.currentWeight += item.weight

		if best == nil || item.currentWeight > best.currentWeight {
			best = item
		}
	}

	if best == nil {
		return nil
	}

	best.currentWeight -= totalWeight
	return best
}

// selectByWeight implements smooth weighted round-robin algorithm
func (s *selector) selectByWeight() *subGroupItem {
	// Keep for backward compatibility or full list selection
	indices := make([]int, len(s.subGroups))
	for i := range s.subGroups {
		indices[i] = i
	}
	return s.selectByWeightFromSubset(indices)
}

// supportsModel checks if subGroupItem supports the model
func (s *selector) supportsModel(item *subGroupItem, modelName string) bool {
	if modelName == "" {
		return true
	}
	if len(item.supportedModels) == 0 {
		return true
	}

	for _, m := range item.supportedModels {
		if m == modelName {
			return true
		}
		if strings.HasSuffix(m, "*") {
			prefix := strings.TrimSuffix(m, "*")
			if strings.HasPrefix(modelName, prefix) {
				return true
			}
		}
	}
	return false
}

// hasActiveKeys checks if a sub-group has available API keys
func (s *selector) hasActiveKeys(groupID uint) bool {
	key := fmt.Sprintf("group:%d:active_keys", groupID)
	length, err := s.store.LLen(key)
	if err != nil {
		logrus.WithFields(logrus.Fields{
			"group_id": groupID,
			"error":    err,
		}).Debug("Error checking active keys, assuming available")
		return true
	}
	return length > 0
}
