// Package affinity defines upstream target bindings and their evictable hot cache.
// The gateway coordinates read-through with the durable BindingStore, backed in
// production by system_settings keys _internal.affinity.binding.<raw-hmac>.
// Hot-cache misses, LRU/capacity eviction and process restarts do not delete
// persisted bindings; durable TTL/capacity cleanup is owned by the storage and
// control layers. Recovery uses the same database and key derivation material.
// Unrelated configuration, catalog/Models.dev updates and snapshot revisions do
// not directly discard bindings, and recovery does not depend on a shutdown
// checkpoint.
//
// A target is the exact upstream route that succeeded: credential identity plus
// the route entry and upstream model. A bound target must pass all current
// access-key, group, route, entry-weight, credential identity and
// cooldown/blacklist checks before being tried first across eligible priority,
// route and regular/store-downgraded buckets; the same credential's sibling
// models stay ordinary candidates. An excluded target retains its durable row
// while ordinary candidates may serve the request. Only an actual retryable
// provider failure or a health bypass of the bound route permits affinity
// fallback and migration after success; filtering or downstream failure does not.
// A remembered route that no longer exists in the snapshot is a miss, so a
// removed or renamed route never locks the key. previous_response_id
// continuation ownership remains a separate mechanism.
//
// Required durable lookup/decode errors fail closed with cache_unavailable before
// provider dispatch. A row written before route identity existed has no entry and
// is a miss, not an error, and is replaced by the next success; there is no
// compatibility layer. All-disabled candidates skip durable lookup. Local
// resolver unavailability and memory-only test setups retain their existing
// boundaries. A durable write error after provider success preserves the
// delivered response and hot binding and is logged; recovery of that failed
// write is not guaranteed.
package affinity
