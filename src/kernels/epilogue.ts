/**
 * Epilogues: an activation applied to a projection's output while it's still
 * in a register, right before the store. Fusing GELU into the MLP's first
 * matmul saves a separate kernel and a full read + write of the (4x wider)
 * hidden activations.
 *
 * Every kernel that supports one includes epilogueWgsl() and wraps its final
 * value in act(...). With "none", act() is the identity and the compiler
 * folds it away.
 */

export type Epilogue = "none" | "gelu";

export function epilogueWgsl(e: Epilogue = "none"): string {
  if (e === "none") return "fn act(x: f32) -> f32 { return x; }";
  // exact GELU via the A&S erf approximation (max error 1.5e-7) - same as
  // the standalone activation kernel, so fused and unfused agree
  return /* wgsl */ `
fn act(x: f32) -> f32 {
  let z = x * 0.7071067811865476;
  let a = abs(z);
  let t = 1.0 / (1.0 + 0.3275911 * a);
  let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
  let erf = sign(z) * (1.0 - poly * exp(-a * a));
  return 0.5 * x * (1.0 + erf);
}`;
}
