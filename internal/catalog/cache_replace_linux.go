//go:build linux

package catalog

import "os"

// replaceCatalogFileAtomic uses same-directory rename, whose replacement is
// atomic on Linux.
func replaceCatalogFileAtomic(temporaryPath, finalPath string) error {
	if err := requireSiblingCatalogPaths(temporaryPath, finalPath); err != nil {
		return err
	}
	return os.Rename(temporaryPath, finalPath)
}
