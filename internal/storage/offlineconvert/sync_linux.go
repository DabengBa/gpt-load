//go:build linux

package offlineconvert

import (
	"golang.org/x/sys/unix"
)

// syncDirectory makes a published directory entry durable. Linux can fsync a
// directory descriptor, which is the only way to persist the link itself.
func syncDirectory(path string) error {
	fd, err := unix.Open(path, unix.O_RDONLY|unix.O_DIRECTORY|unix.O_CLOEXEC, 0)
	if err != nil {
		return err
	}
	defer func() { _ = unix.Close(fd) }()
	return unix.Fsync(fd)
}
