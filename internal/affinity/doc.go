// Package affinity defines upstream target bindings and their evictable hot cache.
// The gateway coordinates read-through with the durable BindingStore, backed in
// production by system_settings keys _internal.affinity.binding.<raw-hmac>.
// Hot-cache misses, TTL/LRU/capacity eviction and process restarts do not delete
// persisted bindings; recovery uses the same database and key derivation material.
// Unrelated configuration, catalog/Models.dev updates and snapshot revisions do
// not discard bindings, and recovery does not depend on a shutdown checkpoint.
//
// A bound target must pass all current access-key, group, route, entry-weight,
// credential identity and cooldown/blacklist checks before being tried first
// across eligible priority, route and regular/store-downgraded buckets. An
// excluded target retains its durable row while ordinary candidates may serve
// the request. Only an actual retryable provider failure permits affinity
// fallback and migration after success; filtering or downstream failure does not.
// previous_response_id continuation ownership remains a separate mechanism.
//
// Required durable lookup/decode errors fail closed with cache_unavailable before
// provider dispatch. All-disabled candidates skip durable lookup. Local resolver
// unavailability and memory-only test setups retain their existing boundaries.
// A durable write error after provider success preserves the delivered response
// and hot binding and is logged; recovery of that failed write is not guaranteed.
package affinity
