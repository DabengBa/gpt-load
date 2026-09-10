//go:build windows

package container

import "go.uber.org/dig"

func provideDebugCapture(*dig.Container) error   { return nil }
func configureDebugCapture(*dig.Container) error { return nil }
