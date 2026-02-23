# Software Release Cycle
- id: release_cycle
- type: documentation
<!-- content -->

## Development
- id: phase.dev
- status: in-progress
- type: documentation
<!-- content -->

### Backend Implementation
- id: dev.backend
- status: done
- type: documentation
<!-- content -->

### Frontend Implementation
- id: dev.frontend
- status: in-progress
- type: documentation
- blocked_by: [dev.backend]
<!-- content -->

## Testing
- id: phase.testing
- status: todo
- type: documentation
- blocked_by: [phase.dev]
<!-- content -->

### Unit Tests
- id: test.unit
- status: todo
- type: documentation
- blocked_by: [dev.backend, dev.frontend]
<!-- content -->

### Integration Tests
- id: test.integration
- status: todo
- type: documentation
- blocked_by: [test.unit]
<!-- content -->

## Deployment
- id: phase.deploy
- status: blocked
- type: documentation
- blocked_by: [phase.testing]
<!-- content -->
