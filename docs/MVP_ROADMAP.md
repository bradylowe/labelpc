# LabelPC MVP Roadmap

## Product Direction

LabelPC is being redesigned as a responsive point-cloud annotation and analysis app. The future architecture is intentionally open: the project may become web-native, keep a Qt engine behind a browser stream for a while, move heavy calculations into Rust/C++/Python workers, or split responsibilities across multiple containers.

The non-negotiable goal is responsiveness with large point-cloud datasets. The app should not blindly render every point in a full-density scan when the user is zoomed out. It should use spatial indexing, level of detail, culling, sampling, preprocessing, and adaptive rendering so users see enough detail for the current task without paying the cost of rendering invisible or indistinguishable points.

## Working Assumptions

- Start with `.las` point-cloud files.
- Add `.laz` and other formats later.
- Run the app through reproducible containers.
- Expect at least a frontend, backend/app engine, and database.
- Store scan metadata, annotation state, shapes, and session data durably.
- Keep local-first operation possible while leaving room for distributed workflows.
- Use pull requests for review.
- Prefer merging rebuild PRs into a development/integration branch until the first release cycle is ready.
- Before overwriting or radically replacing `master`, freeze the current historical `master` state with a branch or tag. The current package version is `4.2.6`, so a freeze name like `v4.2.6` would preserve the old Qt-era codebase before the rebuild.
- Plan a major-version upgrade for the rebuild because compatibility and architecture are expected to change substantially.

## Rebuild and Legacy Code Policy

The rebuild does not need to preserve the current codebase. Existing files, dependencies, packaging, examples, and application modules are disposable if they block the future architecture.

If any inherited code remains, it must be treated as untrusted legacy code until it is reviewed, modernized, tested, and vetted for vulnerabilities. The same applies to dependencies: no existing dependency should be assumed safe or appropriate just because it is already in the repository.

The project may delete most of the existing implementation and still be healthy as long as it preserves the roadmap, product context, release history, and any deliberately retained assets or reference behavior.

### Legacy Retention Rules

- Keep old code only when it clearly accelerates the new architecture.
- Re-vet retained dependencies for security, maintenance status, license fit, and compatibility with containerized deployment.
- Replace or remove vulnerable dependencies where practical.
- Avoid carrying old Qt assumptions into the new app unless they are chosen intentionally.
- Preserve historical `master` before destructive rebuild work lands there.
- Use a development branch as the integration target while the rebuild direction is still evolving.

## MVP 0: Architecture Proof

Goal: prove the chosen stack can launch, load a point cloud, render it interactively, and persist basic scan state before building every annotation tool.

### Scope

- Containerized launch for backend, frontend, and database.
- Responsive frontend shell.
- User event pipeline for mouse move, click, drag, wheel/zoom, and key modifiers.
- Load one `.las` file.
- Generate, store, or use a basic spatial index or level-of-detail representation.
- Render the point cloud from a top-down view.
- Pan and zoom the point cloud.
- Adaptive rendering so zoomed-out views do not brute-force every point.
- Persist scan metadata in the database.
- Display basic point count, visible point count, loaded-file status, and index status.

### Exit Criteria

- A user can launch the app from containers.
- A user can load a representative `.las` file.
- A user can pan and zoom the top-down point-cloud view without obvious lag on the representative dataset.
- The database records that the scan exists.
- The app exposes enough performance/status information to know whether rendering is adaptive.

## MVP 1: Minimal Annotation Loop

Goal: prove the app can create, persist, reload, export, and query annotations.

### Scope

- Drag-box select points and highlight the selection.
- Enter drawing mode.
- Draw one simple annotation shape first: a flat 2D rectangle at `z=0`.
- Select a rectangle.
- Translate and resize the rectangle.
- Assign a basic label, type, color, and identifier to the annotation.
- Save annotation state to the database.
- Reload a saved session.
- Export annotations to JSON.
- Query shape count by shape type.
- Warn about unsaved changes or autosave basic changes.

### Exit Criteria

- A user can load a scan, draw/select/edit one rectangle annotation, save, reload, and verify the annotation is still present.
- A user can export annotation data to JSON.
- A basic query can count shapes by type.

## MVP 2: Product-Shape Expansion

Goal: expand from architecture proof to a credible annotation tool.

### Candidate Scope

- Six axis-aligned 2D views: top, bottom, front, back, left, and right.
- Fit-to-view and reset-camera controls.
- Coordinate and units display.
- Point-cloud selection with visible axes, similar to Unity3D or Blender.
- Translate and rotate the whole point cloud.
- Draw points.
- Draw lines.
- Draw 3D rectangles / boxes with initial thickness and bottom at `z=0`.
- Select, translate, rotate, and resize shapes.
- Export annotations to CSV.
- Add `.laz` support.
- Add more database queries over files, scans, shape types, dimensions, labels, and annotation status.

## Later Product Milestones

These are important, but they should not block the first architecture proof.

- Arbitrary 2D polygons.
- Arbitrary 3D polygons or mesh-like triangular-face annotations.
- Full Blender/Unity-style transform gizmo polish.
- Full widget/plugin framework.
- Distributed multi-user mode.
- Browser streaming or remote viewport delivery if web-native rendering is not the first path.
- Advanced inventory search across all known scans and annotations.
- Support for additional point-cloud formats beyond `.las` and `.laz`.

## Cross-Cutting Requirements

### Performance

- Track frame time, loaded point count, visible point count, selected point count, index status, and import status.
- Prefer screen-space density or similar visible-density rules when zoomed out.
- Increase detail when the user focuses on a smaller region.
- Treat full-density rendering as an explicit high-cost request, not the default interaction model.

### Data

- Store every known scan/file.
- Store scan metadata and provenance.
- Store annotations per scan/file/session.
- Preserve shape geometry, labels, colors, types, and edit history where practical.
- Support future inventory queries across scans and annotations.

### UX

- Keep interaction responsive before adding more tools.
- Provide import status and clear error states for invalid or huge files.
- Provide dirty-state warnings or autosave.
- Provide undo/redo for annotation edits once the first edit loop exists.

### Testing

- Keep small sample data for CI.
- Keep larger representative datasets for manual performance checks.
- Add tests around file import, spatial indexing, annotation persistence, export format, and database queries.
- Track test coverage as the app grows.
