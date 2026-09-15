package control

import (
	"crypto/sha256"
	"encoding/binary"
	"hash"
	"net/textproto"
	"sort"
	"strings"

	"gpt-load/internal/channel"
	"gpt-load/internal/protocol"
	"gpt-load/internal/state"
	"gpt-load/internal/storage/models"
)

// groupProbeSignature binds a manual probe to the complete target definition
// that was tested. It prevents a successful response for an old route from
// restoring a newly configured target.
type groupProbeSignature [sha256.Size]byte

type groupProbeTarget struct {
	protocol  protocol.Protocol
	routeMode channel.RouteMode
	model     string
	signature groupProbeSignature
}

// buildGroupProbeTarget resolves the channel-owned probe contract. Manual
// probes never infer a protocol from the data-plane order and never fall back.
func buildGroupProbeTarget(group state.GroupView, model string) (groupProbeTarget, bool) {
	if strings.TrimSpace(group.ConnectionType) == string(models.ConnectionTypeSubscription) {
		return groupProbeTarget{}, false
	}
	if group.ChannelID == "" || group.ResolvedTarget.ChannelID != group.ChannelID ||
		!group.ResolvedTarget.ProviderKind.Valid() {
		return groupProbeTarget{}, false
	}
	probeModel := strings.TrimSpace(model)
	if probeModel == "" && len(group.Models) > 0 {
		probeModel = strings.TrimSpace(group.Models[0].ID)
	}
	if probeModel == "" {
		return groupProbeTarget{}, false
	}
	probeProtocol, routeMode, ok := group.ResolvedTarget.ProbeRoute(probeModel)
	if !ok || !probeProtocol.SupportsGeneratedText() {
		return groupProbeTarget{}, false
	}
	return groupProbeTarget{
		protocol:  probeProtocol,
		routeMode: routeMode,
		model:     probeModel,
		signature: computeGroupProbeSignature(group, probeProtocol, routeMode, probeModel),
	}, true
}

func computeGroupProbeSignature(
	group state.GroupView,
	probeProtocol protocol.Protocol,
	routeMode channel.RouteMode,
	probeModel string,
) groupProbeSignature {
	hasher := sha256.New()
	writeProbeSignatureUint64(hasher, uint64(group.ID))
	writeProbeSignaturePart(hasher, []byte(group.ChannelID))
	writeProbeSignaturePart(hasher, []byte(group.ResolvedTarget.ProviderKind))
	writeProbeSignaturePart(hasher, group.ResolvedTarget.TargetConfig)
	writeProbeSignaturePart(hasher, []byte(probeProtocol))
	writeProbeSignaturePart(hasher, []byte(routeMode))
	writeProbeSignaturePart(hasher, []byte(probeModel))
	writeProbeSignaturePart(hasher, []byte(group.Proxy.Source))
	writeProbeSignaturePart(hasher, []byte(group.Proxy.Config.Mode))
	writeProbeSignaturePart(hasher, []byte(group.Proxy.Config.URL))

	type headerSetPart struct {
		name  string
		value string
	}
	setParts := make([]headerSetPart, 0, len(group.HeaderRules.Set))
	for name, value := range group.HeaderRules.Set {
		setParts = append(setParts, headerSetPart{
			name: normalizeProbeHeaderName(name), value: value,
		})
	}
	sort.Slice(setParts, func(i, j int) bool {
		if setParts[i].name != setParts[j].name {
			return setParts[i].name < setParts[j].name
		}
		return setParts[i].value < setParts[j].value
	})
	writeProbeSignatureUint64(hasher, uint64(len(setParts)))
	for _, part := range setParts {
		writeProbeSignaturePart(hasher, []byte(part.name))
		writeProbeSignaturePart(hasher, []byte(part.value))
	}

	removeParts := make([]string, len(group.HeaderRules.Remove))
	for index, name := range group.HeaderRules.Remove {
		removeParts[index] = normalizeProbeHeaderName(name)
	}
	sort.Strings(removeParts)
	writeProbeSignatureUint64(hasher, uint64(len(removeParts)))
	for _, name := range removeParts {
		writeProbeSignaturePart(hasher, []byte(name))
	}

	var signature groupProbeSignature
	copy(signature[:], hasher.Sum(nil))
	return signature
}

func writeProbeSignatureUint64(hasher hash.Hash, value uint64) {
	var encoded [8]byte
	binary.BigEndian.PutUint64(encoded[:], value)
	writeProbeSignaturePart(hasher, encoded[:])
}

func writeProbeSignaturePart(hasher hash.Hash, value []byte) {
	var encodedLength [8]byte
	binary.BigEndian.PutUint64(encodedLength[:], uint64(len(value)))
	_, _ = hasher.Write(encodedLength[:])
	_, _ = hasher.Write(value)
}

func normalizeProbeHeaderName(name string) string {
	return strings.ToLower(textproto.CanonicalMIMEHeaderKey(name))
}
