import copy

from cerberus import Validator

from collections import Mapping, Sequence


class ObjectValidator(Validator):
    def __init__(self, *args, **kwargs):
        super(ObjectValidator, self).__init__(*args, **kwargs)
        self.allow_unknown = True

    def validate_object(self, obj):
        return self.validate(obj.__dict__)

    def _validate_type_object(self, value):
        # objects which are not Mapping or Sequence types are allowed.
        # (Mapping and Sequence types are dealt elsewhere.)
        if isinstance(value, object) and \
                not isinstance(value, (Sequence, Mapping)):
            return True

    def _validate_schema(self, schema, field, value):
        if isinstance(value, (Sequence, Mapping)):
            super(ObjectValidator, self)._validate_schema(schema, field, value)
        elif isinstance(value, object):
            validator = copy.copy(self)
            validator.schema = schema
            # validator = self._get_child_validator(document_crumb=field,
            #                                       schema_crumb=(field, 'schema'),
            #                                       schema=schema,
            #                                       allow_unknown=self.allow_unknown)
            if not validator.validate(value.__dict__):
                self._error(validator._errors)