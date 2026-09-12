package affinity

import (
	"bytes"
	"encoding/binary"

	"gpt-load/internal/execution"
	"gpt-load/internal/protocol"
)

const keyDomain = "gpt-load/affinity/signal-hmac/v3"

// SignalType 标识哪个可选请求提示作为亲和缓存键的输入。
type SignalType string

const (
	// SignalPromptPrefix 基于推断的稳定 prompt 前缀推导亲和缓存键。
	SignalPromptPrefix SignalType = "prompt_prefix"
	// SignalPromptCacheKey 基于显式 prompt 缓存键推导亲和缓存键。
	SignalPromptCacheKey SignalType = "prompt_cache_key"
)

func (signal SignalType) Valid() bool {
	return signal == SignalPromptPrefix || signal == SignalPromptCacheKey
}

// Key is an opaque, tenant-scoped affinity cache key.
type Key string

func (key Key) Valid() bool {
	return key != ""
}

// Hasher computes a keyed digest without exposing its key material.
type Hasher interface {
	Hash(string) string
}

// DeriveKey 创建一个限定在单个 access key、协议、
// 客户端模型、操作和请求信号范围内的版本化亲和缓存键。
func DeriveKey(
	hasher Hasher,
	accessKeyID uint,
	clientProtocol protocol.Protocol,
	clientModel string,
	operation execution.Operation,
	signalType SignalType,
	signalValue []byte,
) Key {
	if hasher == nil || accessKeyID == 0 || !clientProtocol.Valid() || !signalType.Valid() || len(signalValue) == 0 {
		return ""
	}
	var material bytes.Buffer
	writeKeyField(&material, []byte(keyDomain))
	var encodedID [8]byte
	binary.BigEndian.PutUint64(encodedID[:], uint64(accessKeyID))
	writeKeyField(&material, encodedID[:])
	writeKeyField(&material, []byte(clientProtocol))
	writeKeyField(&material, []byte(clientModel))
	writeKeyField(&material, []byte(operation))
	writeKeyField(&material, []byte(signalType))
	writeKeyField(&material, signalValue)
	return Key(hasher.Hash(material.String()))
}

func writeKeyField(target *bytes.Buffer, value []byte) {
	var size [8]byte
	binary.BigEndian.PutUint64(size[:], uint64(len(value)))
	target.Write(size[:])
	target.Write(value)
}
