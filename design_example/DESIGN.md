---
name: Monochrome Intelligence
colors:
  surface: '#131313'
  surface-dim: '#131313'
  surface-bright: '#3a3939'
  surface-container-lowest: '#0e0e0e'
  surface-container-low: '#1c1b1b'
  surface-container: '#201f1f'
  surface-container-high: '#2a2a2a'
  surface-container-highest: '#353534'
  on-surface: '#e5e2e1'
  on-surface-variant: '#c4c7c8'
  inverse-surface: '#e5e2e1'
  inverse-on-surface: '#313030'
  outline: '#8e9192'
  outline-variant: '#444748'
  surface-tint: '#c6c6c7'
  primary: '#ffffff'
  on-primary: '#2f3131'
  primary-container: '#e2e2e2'
  on-primary-container: '#636565'
  inverse-primary: '#5d5f5f'
  secondary: '#c8c6c5'
  on-secondary: '#313030'
  secondary-container: '#474746'
  on-secondary-container: '#b7b5b4'
  tertiary: '#ffffff'
  on-tertiary: '#303030'
  tertiary-container: '#e4e2e1'
  on-tertiary-container: '#656464'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#e2e2e2'
  primary-fixed-dim: '#c6c6c7'
  on-primary-fixed: '#1a1c1c'
  on-primary-fixed-variant: '#454747'
  secondary-fixed: '#e5e2e1'
  secondary-fixed-dim: '#c8c6c5'
  on-secondary-fixed: '#1c1b1b'
  on-secondary-fixed-variant: '#474746'
  tertiary-fixed: '#e4e2e1'
  tertiary-fixed-dim: '#c8c6c6'
  on-tertiary-fixed: '#1b1c1c'
  on-tertiary-fixed-variant: '#474747'
  background: '#131313'
  on-background: '#e5e2e1'
  surface-variant: '#353534'
typography:
  headline-lg:
    fontFamily: Geist
    fontSize: 40px
    fontWeight: '600'
    lineHeight: '1.2'
    letterSpacing: -0.02em
  headline-md:
    fontFamily: Geist
    fontSize: 32px
    fontWeight: '600'
    lineHeight: '1.2'
    letterSpacing: -0.01em
  headline-sm:
    fontFamily: Geist
    fontSize: 24px
    fontWeight: '500'
    lineHeight: '1.3'
  body-lg:
    fontFamily: Geist
    fontSize: 18px
    fontWeight: '400'
    lineHeight: '1.6'
  body-md:
    fontFamily: Geist
    fontSize: 16px
    fontWeight: '400'
    lineHeight: '1.6'
  label-md:
    fontFamily: JetBrains Mono
    fontSize: 14px
    fontWeight: '500'
    lineHeight: '1.4'
    letterSpacing: 0.05em
  label-sm:
    fontFamily: JetBrains Mono
    fontSize: 12px
    fontWeight: '500'
    lineHeight: '1.4'
    letterSpacing: 0.08em
  headline-lg-mobile:
    fontFamily: Geist
    fontSize: 32px
    fontWeight: '600'
    lineHeight: '1.2'
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  unit: 4px
  gutter: 24px
  margin-mobile: 16px
  margin-desktop: 64px
  max-width: 1440px
---

## Brand & Style
This design system targets high-stakes professional environments requiring extreme focus and clarity. The brand personality is understated, precise, and high-end, evoking the feeling of "Stealth Intelligence."

The aesthetic merges **Minimalism** with **Glassmorphism**. By stripping away all hues, the UI relies entirely on contrast, texture, and refined typography to communicate hierarchy. It is a tool for experts who value substance over spectacle. The interface should feel like a high-precision instrument: quiet, capable, and expensive.

## Colors
The palette is strictly achromatic, utilizing a range of grays, blacks, and whites to create depth.

- **Stealth (Dark Mode):** The default state. Uses `#000000` for the base background, with `#0F0F0F` and `#1A1A1A` for surface containers. Primary actions are rendered in stark `#FFFFFF` with black text.
- **E-Ink (Light Mode):** Utilizes `#F5F5F5` (off-white) as a base to reduce eye strain. Text and borders use `#121212` (Charcoal) for high legibility without the harshness of pure black.
- **Glass:** Semi-transparent layers are created using white or black with varying alpha channels (e.g., `rgba(255, 255, 255, 0.05)`), allowing background content to subtly bleed through.

## Typography
The typography is centered on technical precision. **Geist** provides a clean, modern sans-serif foundation for high-readability body text and authoritative headlines. 

**JetBrains Mono** is used for labels, data points, and metadata to reinforce the "Intelligence" narrative. Letter spacing for labels is slightly increased to enhance the technical, tabulated feel of the data. Use uppercase for `label-sm` to denote secondary categories.

## Layout & Spacing
The layout follows a strict **Fixed Grid** system on desktop and a fluid system on mobile. 

- **Desktop:** 12-column grid, 1440px max-width, 24px gutters. Content is centered with generous 64px side margins to emphasize focus.
- **Mobile:** 4-column fluid grid with 16px margins. 
- **Rhythm:** All spacing (padding, margins) must be multiples of the 4px base unit. Use larger gaps (48px+) between distinct sections to maintain the minimalist philosophy.

## Elevation & Depth
Depth is communicated through **Glassmorphism** and **Tonal Layering** rather than traditional shadows.

- **Level 0 (Base):** Deepest black or off-white.
- **Level 1 (Surface):** Slightly lighter/darker than base with a subtle 1px border (`#FFFFFF10` in dark mode).
- **Level 2 (Floating):** Semi-transparent background blur (backdrop-filter: blur(12px)) with a more pronounced border.
- **Overlays:** Use high-contrast borders for modals and tooltips. Avoid drop shadows unless they are "Ambient Shadows"—extremely diffused and low opacity (5-10%) to suggest a soft lift without creating visual noise.

## Shapes
Shapes are "Soft" (`0.25rem`). This slight rounding takes the edge off the brutalist monochrome palette, making the high-end interface feel approachable yet engineered. Large containers like cards may use `rounded-lg` (`0.5rem`), but primary buttons and inputs remain at the base `0.25rem` for a crisp, professional appearance.

## Components
- **Buttons:** Primary buttons are Solid White (Dark Mode) or Solid Charcoal (Light Mode). Secondary buttons use a 1px border with no fill. "Ghost" buttons are reserved for tertiary actions.
- **Inputs:** Use a 1px border with a subtle background tint. Focus states are indicated by a weight increase in the border (from 1px to 2px) rather than a color change.
- **Chips:** Monochromatic tags with `label-sm` typography. Backgrounds should be just one step removed from the surface color.
- **Cards:** Defined by 1px borders and subtle tonal shifts. No shadows.
- **Lists:** Separated by thin, low-opacity horizontal rules. Hover states use a slight background highlight (e.g., 5% white overlay).
- **Data Visuals:** Use varying shades of gray or patterns (dots, slashes) to differentiate data sets in charts, maintaining the strictly monochrome constraint.