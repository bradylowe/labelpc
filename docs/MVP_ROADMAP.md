# LabelPC MVP Roadmap

## Product Direction

LabelPC is being redesigned as a responsive point-cloud annotation and analysis app. The default rebuild direction is web-first: if the browser can provide the workflow, performance, and privacy model users need, the primary app should be a web application backed by a durable app engine and database.

The architecture should still leave room for alternate power-user surfaces later. If a future user needs local-machine rendering, privacy, special hardware access, or a workflow that the browser cannot satisfy, a purpose-built client should be able to plug into the same application API as an optional extension. No native framework or renderer is currently endorsed; build the surface that is needed when evidence shows it is needed. Alternate clients are not the initial product default, and they should not own the core data model.

The product is about more than drawing shapes on point clouds. A point cloud is the spatial evidence layer. An annotation is a flexible domain object layered onto that evidence: it may represent a physical thing, a region, a measurement, a class of similar objects, an inspection note, a hypothesis, a workflow state, or an arbitrary nested bundle of user-defined information. Most annotations will have some physical representation that overlaps the point cloud, but the physical representation is only one part of the annotation record.

The non-negotiable goal is responsiveness with large point-cloud datasets. The app should not blindly render every point in a full-density scan when the user is zoomed out. It should use spatial indexing, level of detail, culling, sampling, preprocessing, and adaptive rendering so users see enough detail for the current task without paying the cost of rendering invisible or indistinguishable points.

## Core and Extension Architecture

LabelPC should be modular from the beginning, but the first product should not be a public plugin marketplace. The early extension boundary is an engineering tool: it lets the team swap viewers, renderers, importers, workers, and interaction tools without rewriting the whole app. After the contracts stabilize, the same boundaries may grow into a real external extension system.

The mental model is closer to a narrow, domain-specific version of VS Code than to a monolithic desktop tool. The core owns durable product concepts. Extensions provide optional or replaceable capabilities.

The core system should own:

- Projects, files, scans, sessions, layout state, settings, commands, and undo/redo.
- The application API used by the web frontend and any future native clients.
- The scan registry and durable database migrations.
- Annotation objects, class/type definitions, physical representation records, relationships, provenance, and nested metadata.
- Import/export contracts, query contracts, and permission boundaries.
- The command/event bus that lets viewers, tools, workers, and panels coordinate without directly owning each other.

Replaceable modules may provide:

- Viewer surfaces, such as a top-down web point-cloud viewer, a later 3D web viewer, a local high-performance viewer, or an experimental renderer.
- Interaction tools, such as pan/zoom, drag-select, draw-rectangle, transform, measurement, and shape-edit tools.
- Worker capabilities, such as `.las` import, spatial indexing, level-of-detail generation, geometry calculations, feature extraction, and exports.
- Panels and workflow widgets, such as annotation inspectors, scan inventory views, class/type managers, performance dashboards, and query builders.

Extensions should register capabilities against stable contracts. They should not casually own the database schema, redefine what an annotation means, bypass the core persistence model, or create private state that cannot be queried or exported through the app. The annotation object model stays central even when a particular viewer or tool is replaceable.

For MVP 0, keep the extension surface deliberately small:

- `ViewerProvider`: mounts a viewer, receives scan/camera/input state, renders through its chosen backend, and emits selection or viewport events.
- `ToolProvider`: registers user-facing commands and interaction modes such as pan, drag-select, and draw-rectangle.
- `WorkerProvider`: registers background capabilities such as import, indexing, level-of-detail generation, and export while reporting progress and status.

Everything else should remain boring and central until the team has evidence that another extension point is worth stabilizing.

## Working Assumptions

- Start with `.las` point-cloud files.
- Add `.laz` and other formats later.
- Run the app through reproducible containers.
- Expect at least a frontend, backend/app engine, and database.
- Treat the web app as the primary user surface unless evidence shows it cannot meet the workflow or performance requirements.
- Keep the app API clean enough that future native clients can plug in without forking the product model.
- Use an internal extension registry for early viewers, tools, and workers; postpone any public third-party plugin marketplace.
- Store scan metadata, annotation objects, annotation geometry, class/type definitions, and session data durably.
- Keep local-first operation possible while leaving room for distributed workflows.
- Use pull requests for review.
- Prefer merging rebuild PRs into a development/integration branch until the first release cycle is ready.
- Before overwriting or radically replacing `master`, freeze the current historical `master` state with a branch or tag. The current package version is `4.2.6`, so a freeze name like `v4.2.6` would preserve the old Qt-era codebase before the rebuild.
- Plan a major-version upgrade for the rebuild because compatibility and architecture are expected to change substantially.

## MVP 0 Implementation Decision

This is the starting architecture for MVP 0. It is a decision, not a permanent constraint. Replace pieces later when measurements or product requirements justify it.

### Chosen Stack

- Frontend: TypeScript, React, Vite, deck.gl/luma.gl for the first WebGL point-cloud viewer, and a small local extension registry for `ViewerProvider` and `ToolProvider` implementations.
- Backend/app engine: Python, FastAPI, Pydantic, SQLAlchemy 2, and Alembic.
- Database: PostgreSQL with PostGIS enabled from the beginning.
- Worker model: start with a backend-owned worker process in the same Python codebase and image, coordinated through database-backed job records. Add Redis, Celery, RQ, or a separate queue only when MVP measurements show the local worker is not enough.
- Point-cloud import: `laspy` for `.las` first, with optional `.laz` support through `lazrs` or `laszip` after the basic `.las` path is stable.
- Numeric/indexing layer: NumPy arrays for parsed point data, memory-mapped or chunked derived arrays where useful, and a first simple spatial/index representation that supports both rendering LoD and later region queries.
- Containers: Docker Compose-compatible local development with separate services for frontend, backend/worker, and Postgres/PostGIS. Keep the compose file compatible with Podman where practical, but optimize first for a repeatable local developer path.
- Testing: Pytest for backend/import/index/data-model tests, Vitest for TypeScript contract and utility tests, and Playwright for the first browser smoke tests once the frontend shell exists.

### Rendering and Indexing Direction

Do not make Potree the first implementation. Potree remains a candidate to evaluate, but MVP 0 should avoid making display-only octree data the product's source of truth before selection/query behavior is proven.

The first renderer should use deck.gl's `PointCloudLayer` or a minimal custom deck.gl/luma.gl layer fed by our own derived point buffers. That keeps the frontend path web-native and replaceable while leaving the point-cloud data model under LabelPC control.

The first indexing path should be deliberately simple and owned by the backend:

- Preserve raw file provenance and scan metadata.
- Generate normalized derived point buffers for viewer consumption.
- Generate one simple spatial/LoD structure for top-down rendering and viewport sampling. A coarse tile/grid or octree-like hierarchy is acceptable for MVP 0; pick the smallest implementation that proves adaptive rendering.
- Keep enough point identity or chunk provenance to support MVP 1 drag-box selection and later polygon/region queries.

Long term, LabelPC should support multiple point-cloud representations: raw source data, render-friendly LoD tiles, query-friendly spatial indexes, selected point sets, derived geometry/features, and task-specific indexes. MVP 0 should prove that these representations can be registered and tracked without pretending one structure solves every use case.

### Why This Stack

- React/Vite/TypeScript is the fastest path to a modern browser app shell with reliable interaction tooling and good developer experience.
- deck.gl gives an immediate GPU-backed point-cloud path without locking the product into Potree's conversion/runtime model.
- Python keeps import/index/geometry work close to mature point-cloud and numeric libraries, and it fits the existing project history better than starting the whole backend in a less familiar systems stack.
- FastAPI provides a clear API boundary for the browser and future optional native clients.
- PostgreSQL/PostGIS gives durable relational state, spatial query support, geometry indexes, and JSONB for arbitrary nested annotation metadata.
- SQLAlchemy and Alembic keep database ownership explicit and migration-friendly.
- A database-backed worker avoids early infrastructure sprawl while still matching the future `WorkerProvider` model.

### Intentional Deferrals

- No public plugin marketplace in MVP 0.
- No Potree-first architecture unless a focused spike shows it supports selection/query requirements cleanly.
- No native client, streamed Qt, or VNC/noVNC path unless the web-first proof fails a measured requirement.
- No Redis/Celery/RQ until background work needs concurrency or retry semantics beyond the simple database-backed job model.
- No arbitrary 3D polygon editing before `.las` import, top-down rendering, pan/zoom, and scan persistence work.

### MVP 0 Spike Questions

- Can deck.gl/luma.gl render representative top-down point subsets smoothly with our own LoD buffers?
- How large can the first `.las` import path go before chunking/memory mapping becomes mandatory?
- Which first spatial structure is fastest to implement while preserving future region-selection hooks: grid/tile hierarchy, k-d tree, or octree-like hierarchy?
- Can PostGIS handle the annotation/scan geometry queries we need without storing every raw point as a database row?
- What point identity/provenance should be retained so MVP 1 selection can become durable without exploding storage?

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

## Team Automation

The repository should grow a small agent-team operating model under `docs/roles/`. The first coordinator is the Project Manager, described in `docs/roles/PROJECT_MANAGER.md`.

The intended automation pattern is a single scheduled PM run, likely hourly during active development. The PM reads the roadmap, role docs, priorities, run logs, and blockers; checks the current repo state; delegates bounded next steps to specialist agents when appropriate; records a concise run log; and escalates to Brady when human judgment is needed.

This should start as documentation and status files before becoming a live cron workflow. The PM loop should prove it can coordinate work without creating duplicate branches, noisy updates, or unbounded agent activity.

## MVP 0: Architecture Proof

Goal: prove the chosen stack can launch, load a point cloud, render it interactively, and persist basic scan state before building every annotation tool. MVP 0 should also prove the smallest useful internal extension contracts, not a full plugin marketplace.

### Scope

- Containerized launch for backend, frontend, and database.
- Responsive frontend shell.
- Core app shell with a minimal command/event bus.
- Internal extension registry for the first viewer, first interaction tools, and first import/index worker.
- A first `ViewerProvider` implementation for the top-down point-cloud view.
- First `ToolProvider` implementations for pan, zoom, and basic pointer/drag handling.
- First `WorkerProvider` implementation for `.las` import and index/status reporting.
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
- The first viewer, tool, and worker are registered through internal contracts instead of being hard-wired directly into unrelated layers.

## MVP 1: Minimal Annotation Loop

Goal: prove the app can create, persist, reload, export, and query annotations.

### Scope

- Drag-box select points and highlight the selection.
- Enter drawing mode.
- Draw one simple annotation shape first: a flat 2D rectangle at `z=0`.
- Select a rectangle.
- Translate and resize the rectangle.
- Represent that rectangle as an annotation object whose geometry is only one field of the record.
- Assign a basic label, class/type, color, identifier, and nested metadata payload to the annotation.
- Save annotation state to the database.
- Reload a saved session.
- Export annotations to JSON.
- Query annotation count by class/type and geometry representation type.
- Warn about unsaved changes or autosave basic changes.

### Exit Criteria

- A user can load a scan, draw/select/edit one rectangle annotation, save, reload, and verify the annotation is still present.
- A user can export annotation data to JSON, including label/type data, geometry, and arbitrary nested metadata.
- A basic query can count annotations by class/type and geometry representation type.

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
- Add initial domain classes such as generic shape groups, walls, doors, or other object types that may appear in a scan.
- Add more database queries over files, scans, annotation classes, geometry types, dimensions, labels, nested metadata, and annotation status.

## Annotation Object Model

LabelPC should treat annotation as the broad product concept and geometry as one representation inside it.

An annotation should be able to contain:

- Stable identifier, scan/file/session provenance, timestamps, and edit history.
- Human label, color, workflow status, confidence, owner, and notes.
- Class/type information, including user-defined classes and product-defined classes such as wall, door, fixture, region, measurement, or generic shape group.
- One or more physical representations that overlap the point cloud, such as a rectangle, box, line, point, polygon, mesh, selected point set, derived surface, or linked geometry.
- Arbitrary nested metadata, stored in a way that can survive import/export and evolve without a database migration for every new field.
- Relationships to other annotations, such as group membership, parent/child structures, object parts, alternative interpretations, or references between observations.

The early implementation can keep this simple, but it should not hard-code annotation to mean only "shape." The data model should leave room for annotations that carry thoughts, perspectives, classifications, measurements, and domain-specific object descriptions.

## Later Product Milestones

These are important, but they should not block the first architecture proof.

- Arbitrary 2D polygons.
- Arbitrary 3D polygons or mesh-like triangular-face annotations.
- Full Blender/Unity-style transform gizmo polish.
- Public third-party plugin marketplace or broad external extension SDK.
- Distributed multi-user mode.
- Browser streaming or remote viewport delivery if a web-native viewer cannot satisfy a required workflow.
- Alternate native or local viewer clients, unless evidence shows they are needed earlier for performance, privacy, or hardware access.
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
- Preserve annotation object data, geometry representations, labels, colors, classes/types, nested metadata, relationships, and edit history where practical.
- Support future inventory queries across scans and annotations.

### UX

- Prefer a web-first user experience that is install-light and accessible through the browser.
- Keep future native clients compatible through the app API rather than by forking data semantics.
- Keep interaction responsive before adding more tools.
- Provide import status and clear error states for invalid or huge files.
- Provide dirty-state warnings or autosave.
- Provide undo/redo for annotation edits once the first edit loop exists.

### Testing

- Keep small sample data for CI.
- Keep larger representative datasets for manual performance checks.
- Add tests around file import, spatial indexing, annotation persistence, export format, and database queries.
- Track test coverage as the app grows.
