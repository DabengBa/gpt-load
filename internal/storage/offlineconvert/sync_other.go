//go:build !linux

package offlineconvert

// syncDirectory is a no-op where directory descriptors cannot be fsynced. The
// published link is still atomic; only the extra durability flush is skipped.
func syncDirectory(string) error {
	return nil
}
