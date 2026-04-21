# Design System Strategy: The Intellectual Curator

This document outlines the visual and structural framework for a high-density patent intelligence platform. Our objective is to bridge the gap between the raw utility of a Bloomberg Terminal and the effortless sophistication of a modern editorial publication. 

## 1. Creative North Star: The Intellectual Curator
The "Intellectual Curator" aesthetic moves away from standard SaaS "box-and-line" layouts. Instead, it treats patent data as a curated exhibition. We achieve authority through **Newsreader** (serif) headlines and structural precision through **Manrope** (geometric sans). The system avoids harsh borders in favor of "Tonal Logic," where hierarchy is defined by the weight of the paper and the depth of the stack, not the thickness of a line.

---

## 2. Color & Tonal Architecture
The palette is rooted in a pristine, atmospheric light-mode environment. It prioritizes long-session legibility while using high-end indigo accents to signal professional intelligence.

### Surface Hierarchy & Nesting
To achieve a "bespoke" feel, we abandon the flat grid. We utilize the `surface-container` tokens to create a sense of physical layering:
*   **The Canvas:** Use `background` (#f7f9fc) as the foundational layer.
*   **The Editorial Block:** Use `surface-container-low` (#f2f4f7) for large content areas.
*   **The Highlight Card:** Use `surface-container-lowest` (#ffffff) to make critical data "pop" against the gray background.

### The "No-Line" Rule
**Explicit Instruction:** Do not use 1px solid borders for sectioning or containment. Boundaries must be defined solely through background color shifts. For example, a search sidebar should be `surface-container-high` sitting flush against a `surface` background. The change in hex code provides enough "edge" without the visual clutter of a stroke.

### Signature Gradients
Accents must never be flat. To provide a "soul" to the technical tool, apply a subtle linear gradient to primary actions:
*   **Primary CTA Gradient:** From `primary` (#324cce) to `primary_container` (#4e67e8). 
*   **Direction:** 135-degree angle. This creates a soft, prismatic effect that suggests depth and "active" intelligence.

---

## 3. Typography: The Authoritative Voice
The typographic system uses a triple-font approach to categorize information types instantly.

*   **The Editorial Layer (Newsreader):** Used for `display` and `headline` scales. This serif choice signals academic rigor and legal authority. It turns a patent title into a headline.
*   **The Functional Layer (Manrope):** Used for `title` and `body` scales. Its geometric clarity ensures that interface instructions and long-form descriptions remain legible and modern.
*   **The Technical Layer (Monospace):** While not in the primary scale, all patent IDs, IPC codes, and dates must use a high-legibility monospace font to separate "data" from "narrative."
*   **The Metadata Layer (Inter):** Used for `label` scales to ensure maximum clarity at micro-sizes.

---

## 4. Elevation & Depth
In this design system, depth is a tool for information architecture, not decoration.

*   **Tonal Layering:** Depth is achieved by "stacking" the surface tiers. Place a `surface-container-lowest` card on a `surface-container-low` section to create a soft, natural lift.
*   **Ambient Shadows:** For floating elements (e.g., Command Palettes, Tooltips), use extra-diffused shadows.
    *   *Shadow Config:* `0px 12px 32px rgba(25, 28, 30, 0.06)`. 
    *   The shadow is tinted with the `on-surface` color to feel like natural ambient light.
*   **The "Ghost Border" Fallback:** If a border is required for accessibility (e.g., input fields), use `outline_variant` at **20% opacity**. Never use a 100% opaque border.
*   **Glassmorphism:** Use for persistent navigation rails. Apply a `surface` color with 80% opacity and a `20px` backdrop blur to allow data to bleed through subtly as the user scrolls.

---

## 5. Component Logic

### Buttons
*   **Primary:** Indigo Gradient (Primary to Primary-Container). Text: `on_primary`. 
*   **Secondary:** Ghost-style. No background. `label-md` weight. Uses `primary` for text.
*   **Shape:** Use `md` (0.375rem) for a professional, sharp-but-approachable corner.

### Data Cards
Forbid the use of divider lines. Instead:
1.  Increase vertical white space between sections.
2.  Use a `surface-container-highest` header bar against a `surface-container-lowest` body.
3.  Group related metrics inside a `surface_variant` (#e0e3e6) pod.

### Input Fields
*   **Default:** `surface-container-low` background with a 1px "Ghost Border" (20% `outline_variant`).
*   **Focus:** Transition the border to a 1px solid `primary` and add a subtle `primary_fixed` outer glow (4px blur).

### Chips & Tags
*   For patent classifications, use `secondary_container` with `on_secondary_container` text. 
*   Keep the radius `full` for a distinct "pill" shape that contrasts against the sharp-edged data grid.

---

## 6. Do’s and Don’ts

### Do
*   **Do** embrace high-density layouts. Professionals want data, not empty space. Use whitespace to *organize* data, not to hide it.
*   **Do** use intentional asymmetry. A wide column for the legal abstract paired with a narrow column for metadata creates an editorial feel.
*   **Do** use `on_surface_variant` (#444654) for secondary text to maintain a soft contrast ratio.

### Don't
*   **Don't** use pure black (#000000) or pure white (#FFFFFF) for UI elements unless specifically noted. Use the surface and surface-container tokens to maintain the "off-white" atmosphere.
*   **Don't** use traditional "Drop Shadows" on cards. Rely on the "No-Line" background shifts.
*   **Don't** use standard blue. Always use the indigo-leaning `primary` (#324cce) to ensure the tool feels premium and legal-focused rather than like a generic consumer app.