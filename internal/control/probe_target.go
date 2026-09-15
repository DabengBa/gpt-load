package control

import (
	"crypto/sha256"
	"encoding/binary"
	"hash"
	"net/textproto"
	"sort"
	"strings"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/outboundproxy"
	"gpt-load/internal/protocol"
	"gpt-load/internal/state"
	"gpt-load/internal/storage/models"
)

type credentialDecryptor interface {
	Decrypt(string) (string, error)
}

// groupValidationSignature fingerprints everything a probe result depends on:
// the group identity, resolved provider target, selected probe protocol and
// model, and the outbound proxy/header configuration. A restore proof is only
// valid while this signature is unchanged.
type groupValidationSignature [sha256.Size]byte

// groupValidationTarget is one code-owned probe target: a single protocol, a
// model, the route mode resolved by the channel, and the contract's output
// budget. There is intentionally no protocol fallback list.
type groupValidationTarget struct {
	protocol        protocol.Protocol
	routeMode       channel.RouteMode
	model           string
	maxOutputTokens int
	signature       groupValidationSignature
}

func validationAttemptProxy(
	groupProxy outboundproxy.Effective,
) (outboundproxy.Effective, string, error) {
	effective, err := outboundproxy.NormalizeEffective(groupProxy)
	if err != nil {
		return outboundproxy.Effective{}, "", err
	}
	return effective, "", nil
}

func buildGroupValidationTarget(group state.GroupView) (groupValidationTarget, bool) {
	return buildGroupProbeTarget(group, "")
}

// buildGroupProbeTarget builds a probe target from the channel's code-owned
// probe contract. An empty model selects the group's first configured model;
// the legacy ValidationModel fallback is gone. The channel resolves the route
// mode for the contract protocol and model, so native and converted routes keep
// their declared semantics.
func buildGroupProbeTarget(group state.GroupView, model string) (groupValidationTarget, bool) {
	if strings.TrimSpace(group.ConnectionType) == string(models.ConnectionTypeSubscription) {
		return groupValidationTarget{}, false
	}
	if group.ChannelID == "" || group.ResolvedTarget.ChannelID != group.ChannelID ||
		!group.ResolvedTarget.ProviderKind.Valid() {
		return groupValidationTarget{}, false
	}
	contract, ok := group.ResolvedTarget.ProbeContract()
	if !ok {
		return groupValidationTarget{}, false
	}
	probeModel := strings.TrimSpace(model)
	if probeModel == "" && len(group.Models) > 0 {
		probeModel = strings.TrimSpace(group.Models[0].ID)
	}
	if probeModel == "" {
		return groupValidationTarget{}, false
	}
	routeMode, supported := group.ResolvedTarget.ModeForModel(
		contract.Protocol,
		execution.OperationProbe,
		probeModel,
	)
	if !supported {
		return groupValidationTarget{}, false
	}
	return groupValidationTarget{
		protocol:        contract.Protocol,
		routeMode:       routeMode,
		model:           probeModel,
		maxOutputTokens: contract.MinOutputTokens,
		signature: computeGroupValidationSignature(
			group, contract.Protocol, probeModel, routeMode, contract.MinOutputTokens,
		),
	}, true
}

func computeGroupValidationSignature(
	group state.GroupView,
	selectedProtocol protocol.Protocol,
	probeModel string,
	routeMode channel.RouteMode,
	maxOutputTokens int,
) groupValidationSignature {
	hasher := sha256.New()
	writeValidationSignatureUint64(hasher, uint64(group.ID))
	writeValidationSignaturePart(hasher, []byte(group.ChannelID))
	writeValidationSignaturePart(hasher, []byte(group.ResolvedTarget.ProviderKind))
	writeValidationSignaturePart(hasher, group.ResolvedTarget.TargetConfig)
	writeValidationSignaturePart(hasher, []byte(selectedProtocol))
	writeValidationSignaturePart(hasher, []byte(probeModel))
	// The executed route mode and output budget are part of the probe target;
	// a restore proof must not outlive a change to either.
	writeValidationSignaturePart(hasher, []byte(routeMode))
	writeValidationSignatureUint64(hasher, uint64(maxOutputTokens))
	writeValidationSignaturePart(hasher, []byte(group.Proxy.Source))
	writeValidationSignaturePart(hasher, []byte(group.Proxy.Config.Mode))
	writeValidationSignaturePart(hasher, []byte(group.Proxy.Config.URL))

	type headerSetPart struct {
		name  string
		value string
	}
	setParts := make([]headerSetPart, 0, len(group.HeaderRules.Set))
	for name, value := range group.HeaderRules.Set {
		setParts = append(setParts, headerSetPart{
			name:  normalizeValidationHeaderName(name),
			value: value,
		})
	}
	sort.Slice(setParts, func(i, j int) bool {
		if setParts[i].name != setParts[j].name {
			return setParts[i].name < setParts[j].name
		}
		return setParts[i].value < setParts[j].value
	})
	writeValidationSignatureUint64(hasher, uint64(len(setParts)))
	for _, part := range setParts {
		writeValidationSignaturePart(hasher, []byte(part.name))
		writeValidationSignaturePart(hasher, []byte(part.value))
	}

	removeParts := make([]string, len(group.HeaderRules.Remove))
	for index, name := range group.HeaderRules.Remove {
		removeParts[index] = normalizeValidationHeaderName(name)
	}
	sort.Strings(removeParts)
	writeValidationSignatureUint64(hasher, uint64(len(removeParts)))
	for _, name := range removeParts {
		writeValidationSignaturePart(hasher, []byte(name))
	}

	var signature groupValidationSignature
	copy(signature[:], hasher.Sum(nil))
	return signature
}

func writeValidationSignatureUint64(hasher hash.Hash, value uint64) {
	var encoded [8]byte
	binary.BigEndian.PutUint64(encoded[:], value)
	writeValidationSignaturePart(hasher, encoded[:])
}

func writeValidationSignaturePart(hasher hash.Hash, value []byte) {
	var encodedLength [8]byte
	binary.BigEndian.PutUint64(encodedLength[:], uint64(len(value)))
	_, _ = hasher.Write(encodedLength[:])
	_, _ = hasher.Write(value)
}

func normalizeValidationHeaderName(name string) string {
	return strings.ToLower(textproto.CanonicalMIMEHeaderKey(name))
}
