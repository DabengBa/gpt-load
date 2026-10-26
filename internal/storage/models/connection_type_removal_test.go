package models

import (
	"reflect"
	"testing"
)

func TestGroupHasNoProductConnectionType(t *testing.T) {
	if _, exists := reflect.TypeFor[Group]().FieldByName("ConnectionType"); exists {
		t.Fatal("Group still persists product connection_type")
	}
}
