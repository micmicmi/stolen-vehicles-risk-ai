# Design: stolen-vehicles-project

The locked system. Every Hallmark run reads this first and pages share it. Change it on purpose; the file is the rule.

## System
- Genre: modern-minimal
- Shape of the first page: Long Document (numbered by structure; the report chapters are the spine)
- Theme: Dose (ground: white)
- Axes: light/grotesk/green · posture: Committed (the warm paper and the plastic tints carry the surface; the green stays a signal)

## Tokens
`tokens.css` is the source of truth (extract at `web/tokens.css`; pages inline it because the reports ship as self-contained files). The roles:
```css
:root {
  --color-paper: oklch(97.6% 0.011 88);  --color-paper-2: oklch(96.5% 0.02 160);  --color-paper-3: oklch(93% 0.036 160);
  --color-ink: oklch(21% 0.025 170);  --color-ink-2: oklch(31% 0.02 170);  --color-muted: oklch(49% 0.024 165);
  --color-rule: oklch(86.5% 0.022 140);  --color-rule-2: oklch(70% 0.024 150);
  --color-accent: oklch(50% 0.13 160);  --color-accent-deep: oklch(42% 0.11 162);
  --color-accent-soft: oklch(88% 0.08 160);  --color-accent-ink: oklch(99% 0.006 160);  --color-focus: oklch(45% 0.14 160);
  --p-thu: oklch(86.5% 0.075 160);  --p-wed: oklch(91.5% 0.064 98);  /* the other --p-* tints as needed, never as signal */
  --light: oklch(99.4% 0.016 84);  --glint: oklch(100% 0 0);
  --font-display: "Schibsted Grotesk", "Heebo", ui-sans-serif, sans-serif;
  --font-body: "Instrument Sans", "Heebo", ui-sans-serif, sans-serif;
  --font-outlier: "DM Mono", ui-monospace, monospace;
  --radius-card: 14px;  --radius-inner: 10px;  --radius-pill: 999px;  --rule-hair: 1px;
  --shadow-card: 0 1px 2px oklch(30% 0.03 165 / 0.08), 0 24px 48px -24px oklch(40% 0.08 165 / 0.35);
}
```
The spacing scale, the type scale, the easings and the durations are the scaffold's and live in `tokens.css`.

Hebrew note: the display and body faces carry no Hebrew glyphs; Heebo stands in at 400/500/700 and the Latin faces set the Latin. DM Mono is one role only: identifiers, measurements, label voices.

## Controls
- Primary: green plastic fill (`--color-accent-deep`, glint top, 3px rim, sinks 3px on press) · 10px radius · 52px
- Secondary: clear plastic (same moulded rim) · same radius and height

## Surfaces
- Paper carries a fibre grain (feTurbulence data URI at 5%); the hero carries static window light in `--light`.
- Plastic: a `--p-*` tint at 62% with a glint line and an inner shade; it makes trays, wells and compartments.
- Rules: one hairline for rows and seams; tables and trays lift on `--shadow-card`, never a border and a wide shadow together.
- One ink band per page may reverse out figures with mint labels beneath. Grey, never red or amber, marks what is absent.

## Motion
- Character: still. State only (hover, press). Nothing on scroll; under reduced motion everything ends at once.

## Exports
`web/tokens.css` in this project (the block above, for other pages to inline or link). Tailwind, DTCG or shadcn blocks are added here on request, per `export-formats.md`.
