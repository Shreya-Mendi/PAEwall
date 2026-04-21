# Design System Document: The Sovereign Archive

## 1. Overview & Creative North Star
This design system is engineered for the high-stakes world of patent intelligence. The aesthetic direction—**"The Sovereign Archive"**—merges the brutalist efficiency of a Bloomberg Terminal with the surgical precision of modern engineering tools.

While traditional enterprise software relies on excessive padding and "airy" layouts, this system embraces **Information Density**. We do not fear data; we organize it through a rigorous hierarchy of typography and tonal layering. The experience should feel like a digital vault: heavy, authoritative, and indisputable. It eschews the "playful" trends of the modern web (no glassmorphism, no emojis, no decorative gradients) in favor of a "Structured Editorial" look that prioritizes legibility and technical confidence.

---

## 2. Colors & Surface Architecture
The palette is rooted in a "Deep Canvas" philosophy. By using a near-black base, we reduce eye strain for long-form technical analysis and allow the primary amber to function as a surgical laser, drawing the eye only to what matters.

### Surface Hierarchy & Nesting
To move beyond "standard" UI, we prohibit the use of 1px solid, high-contrast borders for sectioning. Instead, boundaries are defined through **Tonal Layering**. 

*   **The Foundation:** Use `surface` (#121315) as the global canvas.
*   **The Workbench:** Use `surface-container-low` (#1b1c1e) for primary navigation or sidebar zones.
*   **The Intelligence Cell:** Use `surface-container` (#1f2022) for the main content areas.
*   **Active Overlays:** Use `surface-container-high` (#292a2c) for floating panels or active state containers.

### The "Ghost Border" Rule
While we avoid traditional dividers, the density of patent data requires structural clarity. Use "Ghost Borders": 1px strokes using the `outline-variant` token (#524534) at **20% to 40% opacity**. This creates an "etched" look into the dark surface rather than a line drawn on top of it.

---

## 3. Typography: The Editorial Authority
The typographic system uses a tri-font strategy to differentiate between narrative confidence, UI utility, and technical data.

*   **Display & Headlines (Newsreader):** A sophisticated serif used for patent titles and major section headers. This conveys the "Legal/Archive" weight of the data. 
    *   *Usage:* `headline-lg` through `display-sm`.
*   **UI & Navigation (Manrope):** A geometric sans-serif that ensures clarity in high-density menus and controls.
    *   *Usage:* `title-md`, `body-md`.
*   **Technical Identifiers (Inter/Monospace):** For patent numbers (e.g., US-1123456-B2) and IPC codes, use `label-md` or `label-sm` with increased letter-spacing. This ensures every character is distinct.

---

## 4. Elevation & Depth: Tonal Layering
In this design system, depth is a function of light, not shadows.

*   **Layering Principle:** Place `surface-container-lowest` cards on a `surface-container-low` section to create a "recessed" effect. This mimics the look of a physical console where components are seated into a frame.
*   **Ambient Shadows:** Traditional drop shadows are forbidden. If a modal or popover requires separation, use a significantly diffused shadow: `0px 24px 48px rgba(0, 0, 0, 0.5)`. The shadow must feel like an absence of light, not a "glow."
*   **Flat Logic:** Keep all elements on the same perceived optical plane unless they are temporary (tooltips/modals). This maintains the "Bloomberg Terminal" efficiency where everything is accessible at once.

---

## 5. Components & Interaction Patterns

### Buttons: Precision Triggers
*   **Primary:** `primary-container` (#f5a623) background with `on_primary` (#452b00) text. Use `sm` (0.125rem) or `none` (0px) corner radius for a sharper, more professional edge.
*   **Secondary:** No background. Use a `Ghost Border` with `secondary` (#c6c6cb) text. 
*   **Interaction:** On hover, the primary amber should shift slightly to `primary` (#ffc880). No "bounce" animations—use sharp, 150ms linear transitions.

### Data Inputs & Fields
*   **The Integrated Field:** Avoid "boxed" inputs. Use a bottom-border-only approach or a very subtle `surface-container-highest` background. 
*   **States:** Error states must use the `error` (#ffb4ab) token as a 2px left-side accent "status bar" rather than coloring the entire box.

### Lists & Information Chips
*   **Lists:** Forbid the use of divider lines between patent search results. Use `0.5rem` of vertical white space and a subtle background shift (`surface-variant`) on hover to define rows.
*   **Status Chips:** Use small, rectangular tags with `sm` (0.125rem) radius. Functional colors (Green for "Granted", Amber for "Pending", Red for "Expired") must be desaturated to fit the dark canvas, using the `on_secondary_container` (#b7b8bd) for text to ensure readability.

### Technical Workbench (Custom Component)
A multi-pane view where patent text (Serif) sits adjacent to technical drawings (Dark-mode inverted diagrams) and metadata (Monospace/Sans). Panes are separated by a 2px gap using the `surface` color, creating a "grid-gap" look that feels structural.

---

## 6. Do’s and Don’ts

### Do
*   **Embrace Density:** It is okay to have 20+ data points on screen if they are hierarchically sorted using `title-sm` and `label-sm`.
*   **Use Intentional Asymmetry:** In the dashboard, align heavy technical data to the left and high-level SERIF headlines to the right to create a "Technical Journal" layout.
*   **Trust the Amber:** Use `#F5A623` sparingly. It is a beacon for action, not a decorative element.

### Don't
*   **No Glassmorphism:** Never use background blurs or transparency that obscures data. The UI must be opaque and readable.
*   **No Gradients:** Colors must be flat and purposeful. We rely on the contrast between `surface-container` tiers for depth.
*   **No Rounded Corners over 8px:** High-end technical tools feel "soft" and "consumer-grade" when corners are too round. Stick to `DEFAULT` (0.25rem) or `sm` (0.125rem).