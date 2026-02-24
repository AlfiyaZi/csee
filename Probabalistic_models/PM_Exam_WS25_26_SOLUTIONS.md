# PM Exam WS 2025/26 -- Full Detailed Solutions

**Result: 36/38 (94.74%)**
**Mistakes: Q2, Q13**

---

## SECTION 1: Distributions and BNs (Questions 1--10)

### The Model (inferred from questions)

"Slightly extended model of the climate in Death Valley" with 4 Boolean variables:
- **W** = WetGround
- **M** = Mosquitos
- **F** = Flowers
- **R** = Rain

**Structure**:

```
    W
   / \
  v   v
  M   F
   \ /
    v
    R
```

Edges: W->M, W->F, M->R, F->R

**BN Factorisation**: P(W, M, F, R) = P(W) * P(M|W) * P(F|W) * P(R|M,F)

This means:
- W is the root (no parents)
- M and F are children of W (common cause / fork at W)
- R is a child of both M and F (v-structure / collider at R)

---

### Question 1 -- CORRECT
> **"M and F are independent."** --> **FALSE**

**Theory**: Two variables are marginally independent iff there is no active (unblocked) path between them with empty evidence set.

**Analysis**:
Path M <-- W --> F is a **fork** (common cause). With no evidence, forks are **active** (unblocked). Information flows from M to F through their common parent W.

Therefore M and F are **dependent** (marginally correlated through W). If it rains, ground gets wet, which promotes both mosquitos and flowers -- knowing about mosquitos tells us something about wetness, which tells us about flowers.

**Compare WS24/25 Q5**: "(Starter_broken _|_ Battery_dead)" was TRUE because those variables had no active path in that graph. Here, M and F DO have an active path through W.

---

### Question 2 -- INCORRECT (student got this wrong)
> **"The full joint distribution contains only 9 non-redundant parameters."** --> **TRUE**

**Theory**: In a BN with Boolean variables, each variable Xi with k Boolean parents has **2^k independent parameters** (one free parameter per row of the CPT, since each row of 2 values sums to 1).

**Calculation**:

| Variable | Parents | # parent configs | Independent params |
|----------|---------|------------------|--------------------|
| W | none | 1 | 1 |
| M | W | 2 | 2 |
| F | W | 2 | 2 |
| R | M, F | 4 | 4 |
| **Total** | | | **9** |

The full joint over 4 Boolean variables would normally have 2^4 - 1 = 15 free parameters. But this BN's independence assumptions reduce it to only **9**. The BN is a compact representation.

**Why the student probably got it wrong**: Likely confused "full joint distribution" (15 params) with "BN parameterisation" (9 params). The key insight is that THIS MODEL (with these independence constraints) can only represent distributions from a 9-dimensional subspace. The model's full joint IS fully specified by 9 parameters.

**Compare WS24/25 Q9**: Similar parameter-counting question for the car diagnosis network. Same technique: sum up CPT sizes for each node.
**Compare WS24/25 Q26**: "How many independent parameters for variable I?" = (|I|-1) * |D| * |S| * |G| * |L| = 1 * 2 * 2 * 3 * 2 = 24.

---

### Question 3 -- CORRECT
> **"If both M and F are false, then W=true can change my degree of belief about R."** --> **FALSE**

**Theory**: This asks whether W and R are d-separated given {M=false, F=false}.

**Analysis**:
R's parents are M and F. If we observe M=false and F=false, then R is completely determined by P(R | M=false, F=false) -- a specific row of R's CPT. Knowing W adds nothing because:

Path W --> M --> R: M is in evidence (observed), so this chain is **blocked**.
Path W --> F --> R: F is in evidence (observed), so this chain is **blocked**.

R _|_ W | {M, F}. Once we know M and F, W gives no additional information about R.

**Intuition**: Rain depends on mosquitos and flowers. If we already know whether there are mosquitos and flowers, knowing whether the ground is wet adds nothing about rain.

---

### Question 4 -- CORRECT
> **"If we reversed the direction of the arrow F -> R, the number of parameters in the model remains the same."** --> **TRUE**

**Theory**: Reversing an edge in a BN changes the parents of two nodes. The total parameter count depends only on the number of parents each node has.

**Analysis**:
- **Original**: W->M, W->F, M->R, F->R
  - P(W): 1, P(M|W): 2, P(F|W): 2, P(R|M,F): 4. **Total = 9**
- **After reversing F->R to R->F**: W->M, W->F, R->F, M->R. Wait -- F now has parents {W, R} and R has parent {M}.
  Actually: the edge F->R becomes R->F. So R loses parent F, and F gains parent R.
  - R's parents: {M} (only M now). P(R|M): 2 params
  - F's parents: {W, R} (W from original + R from reversal). P(F|W,R): 4 params
  - P(W): 1, P(M|W): 2, P(F|W,R): 4, P(R|M): 2. **Total = 9**

Same total! Reversing an edge between two nodes X->Y just moves one "parent slot" from Y's CPT to X's CPT. If both nodes already share the same other parents, the redistribution is exactly balanced.

**General principle**: Reversing a single edge in a BN between nodes where the CPTs involve the same set of other variables keeps the total parameter count the same.

---

### Question 5 -- CORRECT
> **P(W, M, F, R) = P(W, M, F) * P(R|M,F)** --> **TRUE**

**Theory**: By the product rule of probability: P(A, B) = P(A) * P(B|A). This is ALWAYS true, regardless of BN structure -- it's a basic probability identity.

**Analysis**:
P(W, M, F, R) = P(W, M, F) * P(R | W, M, F)

But in this BN, R _|_ W | {M, F} (as shown in Q3). Therefore:
P(R | W, M, F) = P(R | M, F)

So: P(W, M, F, R) = P(W, M, F) * P(R | M, F) ✓

This also follows directly from the BN chain rule:
P(W) * P(M|W) * P(F|W) * P(R|M,F) = [P(W) * P(M|W) * P(F|W)] * P(R|M,F) = P(W,M,F) * P(R|M,F)

---

### Question 6 -- CORRECT
> **"It is possible to parametrise this model such that M and F are perfectly correlated."** --> **TRUE**

**Theory**: A BN structure constrains which independencies MUST hold, but it doesn't prevent strong correlations between dependent variables.

**Analysis**:
M and F share common parent W. We can set:
- P(M=true | W=true) = 1, P(M=true | W=false) = 0
- P(F=true | W=true) = 1, P(F=true | W=false) = 0

Result: M = W and F = W, so M = F always. Perfect correlation achieved!

**Key insight**: BN structure only forces independence constraints, never forces variables to be independent when there IS a connecting path. If variables are connected, the CPT values can make them as correlated as desired.

---

### Question 7 -- CORRECT
> **"Learning about the value of R cannot change my degree of belief regarding W."** --> **FALSE**

**Theory**: This asks if W _|_ R (marginal independence with no evidence).

**Analysis**:
Paths from W to R:
1. W --> M --> R (chain, no evidence nodes in between -> **active**)
2. W --> F --> R (chain, no evidence nodes in between -> **active**)

Both paths are active! W and R are **dependent**. Observing R DOES change our belief about W.

**Intuition**: Rain (R) depends on mosquitos and flowers, which depend on wet ground. If we learn it rained, we infer mosquitos/flowers were likely present, which means the ground was likely wet.

**Compare WS24/25 Q7**: "Learning about R cannot change belief about W" -- same type of question about whether observation of a variable that is connected through a chain can influence belief about a distant ancestor. The answer is always NO for d-separation questions where the path is active.

---

### Part b: All edges removed (Q8--Q10)

Now consider the same 4 variables but with **no edges at all** -- 4 disconnected nodes.

Factorisation: P(W, M, F, R) = P(W) * P(M) * P(F) * P(R)

All variables are **mutually independent**.

---

### Question 8 -- CORRECT
> **"A model with this structure cannot encode a correlation between M and F."** --> **TRUE**

**Theory**: In a BN with no edges, the joint factorises as a product of marginals. This forces all variables to be mutually independent.

**Analysis**:
P(M, F) = P(M) * P(F) (by the BN factorisation with no edges)

This IS the definition of independence. No matter what values we choose for P(M) and P(F), they will always be independent. The only way M and F can be "perfectly correlated" is the degenerate case P(M=true)=P(F=true)=1 or =0, but even then Cov(M,F)=0.

**Contrast with Q6**: With edges (W->M, W->F), we COULD encode perfect correlation. Without edges, we cannot.

---

### Question 9 -- CORRECT
> **"In this model, for any probabilistic query, P(X|e) = P(X)."** (ignoring e=x) --> **TRUE**

**Theory**: When all variables are independent, observing any subset of variables gives no information about the others.

**Analysis**:
Since P(X1,...,Xn) = P(X1)*...*P(Xn), for any variable X and evidence e involving other variables:
P(X | e) = P(X, e) / P(e) = P(X) * P(e) / P(e) = P(X)

Evidence is irrelevant. Every conditional equals the prior.

---

### Question 10 -- CORRECT
> **"In this model, it does not matter which parameters we choose because the variables cannot influence each other."** --> **FALSE**

**Theory**: Even though variables are independent, each individual variable's marginal distribution still matters.

**Analysis**:
The parameters still determine, for example:
- P(Rain = true) -- could be 0.01 (desert) or 0.99 (tropical)
- P(Mosquitos = true) -- could be 0.5 or 0.001

These marginals change the answers to queries like P(R=true) = ? The variables don't influence EACH OTHER, but the parameters still define each variable's own distribution. Choosing different parameters gives different models that make different predictions.

---

## SECTION 2: Inference in BNs (Questions 11--17)

### The Model (inferred from questions)

A different Death Valley model structure:

```
R --> W --> M
      |
      v
      F
```

Edges: R->W, W->M, W->F

**BN Factorisation**: P(R, W, M, F) = P(R) * P(W|R) * P(M|W) * P(F|W)

- R is root (Rain causes WetGround)
- W is child of R
- M and F are children of W (WetGround causes Mosquitos and Flowers)

---

### Part a: Exact Inference

### Question 11 -- CORRECT
> **"In answering the query P(W|F), we need to sum over R before we sum over M."** --> **FALSE**

**Theory**: Variable Elimination lets us choose ANY elimination ordering for hidden variables. Some orderings are more efficient, but ALL orderings give the correct answer.

**Analysis**:
P(W | F) = alpha * SUM_R SUM_M P(R) * P(W|R) * P(M|W) * P(F|W)

We need to eliminate hidden variables R and M. Let's check both orderings:

**Order 1: Eliminate M first, then R**
= alpha * SUM_R P(R) * P(W|R) * P(F|W) * [SUM_M P(M|W)]
= alpha * SUM_R P(R) * P(W|R) * P(F|W) * 1    (SUM_M P(M|W) = 1)
= alpha * P(F|W) * SUM_R P(R) * P(W|R)
= alpha * P(F|W) * P(W)

**Order 2: Eliminate R first, then M**
= alpha * SUM_M P(M|W) * P(F|W) * [SUM_R P(R) * P(W|R)]
= alpha * P(F|W) * P(W) * [SUM_M P(M|W)]
= alpha * P(F|W) * P(W)

Both orderings work! We do NOT "need" to sum over R before M. In fact, eliminating M first is trivial since SUM_M P(M|W) = 1.

**Compare WS24/25 Q11**: Similar question about elimination ordering. The key principle is that VE works with ANY ordering -- some are just more efficient.

---

### Question 12 -- CORRECT
> **"Any query that involves M and/or F as evidence requires evidential reasoning."** --> **TRUE**

**Theory**: There are three types of reasoning in BNs:
1. **Causal** (top-down): from causes to effects (following edge directions)
2. **Evidential** (bottom-up): from effects to causes (against edge directions)
3. **Intercausal** (explaining away): between causes of a common effect

**Analysis**:
In this model, M and F are **leaf nodes** (children of W, no children of their own). They are at the bottom of the causal chain R->W->{M,F}.

Any query with M or F as evidence means we observe an EFFECT and reason about its CAUSE (W). This is by definition **evidential reasoning** -- reasoning against the direction of edges.

Example: P(W | m) requires reasoning from M (effect) up to W (cause).

(Exception noted: pathological cases like P(M|M) are excluded per the question.)

---

### Question 13 -- INCORRECT (student got this wrong)
> **"To answer the query P(F|not_w), we need not consider R."** --> **TRUE**

**Theory**: If a variable is d-separated from the query variable given the evidence, it can be ignored (its contribution cancels out).

**Analysis**:
Query: P(F | W=false)

In this BN, F's only parent is W. Given W (observed as evidence), F is **conditionally independent** of everything else:

F _|_ R | W (because R->W->F is a chain, and W is observed -> blocked)

Therefore:
P(F | not_w) = P(F | W=false)

This is just read directly from F's CPT! We look up the row where W=false and read P(F | W=false). R is completely irrelevant.

**Mathematical proof**:
P(F | not_w) = alpha * SUM_R SUM_M P(R) * P(not_w | R) * P(M | not_w) * P(F | not_w)
= alpha * P(F | not_w) * SUM_R P(R) * P(not_w | R) * SUM_M P(M | not_w)
= alpha * P(F | not_w) * P(not_w) * 1
= P(F | not_w)  after normalisation

R appears only inside a sum that produces a constant (P(not_w)), which gets absorbed by the normalisation. **R can be ignored**.

**Why the student probably got it wrong**: May have thought that since R influences W (and W is in the query), we need to account for R. But since W is EVIDENCE (observed), not a query variable, R's influence on W is irrelevant -- W's value is already known.

**Compare WS24/25 Q13**: "P(query | evidence) -- do we need to consider variable X?" Key principle: if X is d-separated from the query variable given evidence, it can be dropped.

---

### Question 14 -- CORRECT
> **P(f | w, r) can never be larger than P(f | w)** --> **TRUE**

**Theory**: If X _|_ Y | Z, then P(X | Y, Z) = P(X | Z). "Can never be larger" means P(f|w,r) <= P(f|w).

**Analysis**:
In this BN: F _|_ R | W (as established above -- the path R->W->F is blocked when W is observed).

Therefore: P(f | w, r) = P(f | w) for ALL values of r.

Since they are **always equal**, P(f|w,r) can never be larger than P(f|w). It can never be smaller either -- they're the same number!

The statement "can never be larger" is TRUE (it's always exactly equal).

---

### Part b: Approximate Inference

**Specific probability tables given**:
- P(R): P(R=true) = 0, P(R=false) = 1
- P(W|R): P(W=t|R=t) = 1, P(W=f|R=t) = 0, P(W=t|R=f) = 0.1, P(W=f|R=f) = 0.9

So Rain NEVER occurs (P(R=true) = 0 exactly).

---

### Question 15 -- CORRECT
> **"The Forward Sampling algorithm cannot produce a single sample with Rain=true."** --> **TRUE**

**Theory**: In Forward Sampling, we sample variables in topological order from their conditional distributions given already-sampled parents. Root nodes are sampled from their prior.

**Analysis**:
R is a root node. Forward sampling samples R from P(R):
- P(R=true) = 0
- P(R=false) = 1

Since P(R=true) = 0 **exactly**, the sampling process will NEVER generate R=true. It's impossible, not just unlikely.

This is different from a case like P(R=true) = 0.0001 where it would be very rare but possible. Here it's exactly zero.

**Contrast with Q28 (HMM section)**: "Robot on for 1000 steps has probability zero" is FALSE because P(on->on) = 0.8 > 0, so 0.8^999 > 0. Here, P(R=true) = 0 exactly, so no power of transitions can help.

---

### Question 16 -- CORRECT
> **"In Gibbs Sampling, resampling R can always be done independently of the values of M and F."** --> **TRUE**

**Theory**: In Gibbs Sampling, each non-evidence variable is resampled from its distribution conditioned on its **Markov Blanket** (parents + children + co-parents of children).

**Analysis**:
R's Markov Blanket in the graph R->W->{M,F}:
- Parents(R) = {} (none -- R is root)
- Children(R) = {W}
- Co-parents of W = {} (W's only parent is R)

**MB(R) = {W}**

When resampling R in Gibbs, we compute P(R | MB(R)) = P(R | W). This depends ONLY on W, not on M or F.

Therefore, resampling R is always independent of M and F. ✓

**Compare WS24/25 Q17**: "No variable has Markov blanket > 5 variables" -- that was about counting MB size. Here it's about which specific variables are in the MB.

---

### Question 17 -- CORRECT
> **"Rejection Sampling for the query P(R|m,f) cannot return P(R=true|m,f) > 0."** --> **TRUE**

**Theory**: Rejection Sampling generates forward samples and discards those inconsistent with evidence. The estimated probability is the frequency in kept samples.

**Analysis**:
Step 1: Generate forward samples. Since P(R=true) = 0, ALL forward samples have R=false.
Step 2: Keep only samples consistent with evidence (M=m, F=f). All kept samples still have R=false.
Step 3: Estimate P(R=true | m, f) = count(R=true in kept samples) / count(kept samples) = 0/N = 0.

No sample will ever have R=true, so the estimate will ALWAYS be exactly 0, never > 0.

**Note**: This is actually correct! Since P(R=true) = 0, the true posterior P(R=true | anything) = 0 as well. So rejection sampling gives the correct answer here (unlike many cases where it's just an approximation).

---

## SECTION 3: Parameter Learning (Questions 18--23)

**Setup**: Thumbtack with theta = P(heads). Dataset D of N tosses, exactly k = N/2 are heads (perfectly balanced).

**Likelihood function**: L(theta; D) = theta^k * (1-theta)^(N-k) = theta^(N/2) * (1-theta)^(N/2)

**MLE**: theta_hat = k/N = (N/2)/N = **0.5**

---

### Question 18 -- CORRECT
> **"Bayesian estimation with a uniform (constant) prior will lead to a posterior distribution function that is identical to the likelihood function."** --> **FALSE**

**Theory**:
- Posterior proportional to Likelihood * Prior: P(theta|D) ∝ L(theta;D) * P(theta)
- Uniform prior: P(theta) = Beta(1,1) = 1 for theta in [0,1]
- Posterior ∝ L(theta;D) * 1 = L(theta;D)

So posterior is **proportional** to likelihood, but NOT **identical**!

**The difference**: The likelihood L(theta;D) is NOT a proper probability distribution -- it doesn't integrate to 1. The posterior must integrate to 1 (it's a proper distribution). So:

P(theta|D) = L(theta;D) / integral_0^1 L(theta;D) d_theta

The posterior has the same SHAPE as the likelihood but is RESCALED. They are proportional but not identical as functions.

**Also**: With a Beta(1,1) prior: Posterior = Beta(1 + N/2, 1 + N/2), while the likelihood shape corresponds to Beta(N/2 + 1, N/2 + 1) -- wait, let me be precise:
- Likelihood ∝ theta^(N/2) * (1-theta)^(N/2) -- this has the shape of Beta(N/2+1, N/2+1) but isn't normalised
- Posterior = Beta(1+N/2, 1+N/2) -- this IS a proper normalised Beta distribution

They have the same shape but are NOT the same function (different scaling).

**Compare WS24/25 Q18-21**: Similar questions about likelihood function properties. The key is understanding that likelihood is NOT a probability distribution over theta.

---

### Question 19 -- CORRECT
> **L(0.5; D) = 1** --> **FALSE**

**Theory**: The likelihood at theta=0.5 is: L(0.5; D) = 0.5^(N/2) * 0.5^(N/2) = 0.5^N

**Calculation**:
- For N=2: L(0.5) = 0.5^2 = 0.25
- For N=10: L(0.5) = 0.5^10 ≈ 0.001
- For any N > 0: L(0.5) = 0.5^N < 1

The likelihood at the MLE is 0.5^N, which is always **less than 1** for any positive N.

**Common misconception**: Students sometimes think "if theta=0.5 is the MLE, then L(0.5)=1." No! The MLE is the MAXIMISER of L, but the maximum VALUE of L is not necessarily 1. The likelihood function is NOT a probability distribution over theta.

---

### Question 20 -- CORRECT
> **"Laplace smoothing will not change the MLE theta_hat, regardless of alpha."** --> **TRUE**

**Theory**: Laplace smoothing for a Bernoulli parameter:
theta_Laplace = (k + alpha) / (N + 2*alpha)

**Calculation** (with k = N/2):
theta_Laplace = (N/2 + alpha) / (N + 2*alpha) = (N + 2*alpha) / (2*(N + 2*alpha)) = **1/2 = 0.5**

Remarkable! When the data is perfectly balanced (k = N/2), adding the SAME pseudocount alpha to both numerator and denominator doesn't change the ratio. It's still exactly 0.5 regardless of alpha.

**Intuition**: Laplace smoothing adds alpha "virtual heads" and alpha "virtual tails". If the real data already has equal heads and tails, adding equal pseudocounts to both sides keeps the balance.

**Contrast**: If k != N/2 (unbalanced data), Laplace WOULD change the estimate, pulling it toward 0.5.

---

### Question 21 -- CORRECT
> **"The likelihood function L(theta;D) is zero at theta=0 and theta=1."** --> **TRUE**

**Theory**: L(theta) = theta^(N/2) * (1-theta)^(N/2)

**Calculation**:
- L(0) = 0^(N/2) * 1^(N/2) = 0 * 1 = **0** (since N/2 > 0, meaning k > 0, so we saw at least one head)
- L(1) = 1^(N/2) * 0^(N/2) = 1 * 0 = **0** (since N-k = N/2 > 0, so we saw at least one tail)

**Intuition**: theta=0 means "heads is impossible" but we observed heads -> probability of data is 0. Similarly, theta=1 means "tails is impossible" but we observed tails -> probability of data is 0.

---

### Question 22 -- CORRECT
> **"The likelihood function is exactly symmetric."** --> **TRUE**

**Theory**: A function f(theta) is symmetric around 0.5 iff f(theta) = f(1-theta) for all theta.

**Proof**:
L(theta) = theta^(N/2) * (1-theta)^(N/2)
L(1-theta) = (1-theta)^(N/2) * (1-(1-theta))^(N/2) = (1-theta)^(N/2) * theta^(N/2) = L(theta) ✓

The likelihood is symmetric because k = N-k = N/2 (equal exponents). The function is a mirror image around theta=0.5.

**Contrast**: If k != N-k (e.g., 3 heads out of 5), the likelihood theta^3*(1-theta)^2 is NOT symmetric.

---

### Question 23 -- CORRECT
> **"For k = N/2, the posterior P(theta|D) will have its maximum at 0.5, regardless of the prior."** --> **FALSE**

**Theory**: Posterior ∝ Likelihood * Prior. The posterior's mode depends on BOTH the likelihood and the prior.

**Analysis**:
The likelihood is symmetric with maximum at 0.5. But the prior can be arbitrary!

**Counterexample**: Choose a prior Beta(10, 1):
- Prior has mode at theta = (10-1)/(10+1-2) = 9/8... actually let's use a concrete example.
- Prior that is very concentrated near theta=0.9

Then: Posterior ∝ theta^(N/2) * (1-theta)^(N/2) * Prior(theta)

For small N, the prior dominates and the posterior mode can be far from 0.5.

For example, with N=2, k=1:
- Likelihood = theta * (1-theta), max at 0.5
- Prior = Beta(100, 1) ∝ theta^99, concentrated near 1
- Posterior ∝ theta^100 * (1-theta), max near theta = 100/101 ≈ 0.99

So the posterior mode is NOT at 0.5. The statement is FALSE.

**Key insight**: Only a symmetric prior (symmetric around 0.5) would keep the posterior mode at 0.5 when combined with a symmetric likelihood.

---

## SECTION 4: Structure Learning (Questions 24--27)

**Setup**: Learning from dataset D of N atomic events over {R, W, M, F} (Boolean).
Two competing structures G1 (top) and G2 (bottom). From context, G1 likely has more edges than G2 (or different topology).

**BIC Score**: BIC(G) = LL(theta_MLE; D, G) - (Dim[G]/2) * log(N)

---

### Question 24 -- CORRECT
> **"If G1 and G2 have the same Maximum Likelihood score, the BIC score would favour G1."** --> **FALSE**

**Theory**: BIC = LL - (d/2)*log(N). If LL is the same for both, BIC difference = -(d1/2)*log(N) + (d2/2)*log(N) = (d2-d1)/2 * log(N).

**Analysis**:
BIC favours the model with **fewer parameters** (smaller d) when LL is tied. The penalty term -(d/2)*log(N) is more negative for larger d.

Since the answer is FALSE, G1 must have **more parameters** (or at least not fewer) than G2. With equal LL, BIC would favour G2 (the simpler model), not G1.

**Principle**: BIC implements Occam's razor -- when models fit equally well, prefer the simpler one.

**Compare WS24/25 Q27-28**: Same principle. BIC can go either way depending on whether the likelihood gain outweighs the complexity penalty.

---

### Question 25 -- CORRECT
> **"If we doubled the training set D by duplicating all examples, the Maximum Likelihood score for both models would increase."** --> **FALSE**

**Theory**: "Maximum Likelihood score" typically means the log-likelihood: LL(theta_MLE; D) = SUM_i log P(xi | theta_MLE).

**Analysis**:
If we duplicate every example, N' = 2N, but the empirical distribution stays the same. So theta_MLE doesn't change.

LL_new = SUM_{i=1}^{2N} log P(xi | theta_MLE) = 2 * SUM_{i=1}^{N} log P(xi | theta_MLE) = 2 * LL_old

Since each log P(xi | theta) <= 0 (log of a probability), LL_old < 0. Therefore 2 * LL_old < LL_old.

The ML score **decreases** (becomes more negative), not increases!

**Intuition**: Each data point's log-probability is negative. Doubling the data doubles the sum of negative numbers -> more negative total. The MLE parameters don't change because the empirical proportions are the same.

**If "ML score" means the likelihood (not log-likelihood)**: L_new = L_old^2. Since 0 < L_old < 1, L_old^2 < L_old. Still decreases!

---

### Question 26 -- CORRECT
> **"Any model structure G with five edges would have a higher Dim[G] in the BIC score than both G1 and G2."** --> **TRUE**

**Theory**: In a BN over Boolean variables, each edge adds parameters. More edges = more parents for some nodes = larger CPTs = more parameters.

**Analysis**:
For 4 Boolean variables, the maximum number of edges is 4*3/2 = 6 (complete DAG). G1 and G2 have fewer than 5 edges each (since 5 edges out of 4 variables is nearly complete).

Actually with 4 variables, a DAG can have at most 4+3+2+1-... well, max edges in a DAG on 4 nodes = C(4,2) = 6. 5 edges means only 1 edge away from complete.

Any graph with 5 edges has more edges than both G1 and G2, and since adding edges to a BN always adds parameters (or keeps them the same), Dim[G_5edges] >= Dim[G1] and Dim[G_5edges] >= Dim[G2].

Given that both G1 and G2 have distinctly fewer than 5 edges, the dimension is strictly larger. ✓

---

### Question 27 -- CORRECT
> **"G1 cannot have a higher Maximum Likelihood score than G2 because it is (partly) anti-causal."** --> **FALSE**

**Theory**: The Maximum Likelihood score of a BN structure depends only on the **statistical fit** to the data, not on whether edges follow causal directions.

**Analysis**:
The ML scoring function doesn't care about causality. A model R->W (rain causes wet ground) and W->R (wet ground causes rain) can have different ML scores, but neither is automatically higher or lower just because of causal direction.

The causal direction is a semantic interpretation. Statistically, what matters is whether the structure's independence constraints match the data's correlations.

**Principle**: ML scoring is purely statistical. "Anti-causal" structures can fit data just as well (or better) than causal ones. Model selection based on BIC or other scores considers fit + complexity, not causality.

---

## SECTION 5: Hidden Markov Models (Questions 28--33)

### The Model: Vacuum Cleaner Robot

**States**: S = {on, off}
**Observations**: O = {squeak, rattle, none}

**Parameters**:
- **Pi** = [0.5, 0.5] (equal probability of starting on or off)
- **A** (transition matrix):
  ```
          to_on  to_off
  from_on  [0.8    0.2]
  from_off [0.5    0.5]
  ```
- **B** (emission matrix):
  ```
             squeak  rattle  none
  on         [0.5     0.4    0.1]
  off        [0.0     0.0    1.0]
  ```

Key observations from B:
- When on: robot makes squeak (50%), rattle (40%), or none (10%)
- When off: robot ALWAYS makes no noise (none = 100%)
- **If we hear squeak or rattle, the robot is DEFINITELY on** (B(off, squeak) = B(off, rattle) = 0)
- If we hear none, robot could be on (10%) or off (100%)

---

### Question 28 -- CORRECT
> **"The probability that the robot is on for 1000 time steps is zero."** --> **FALSE**

**Theory**: P(on for 1000 steps) = P(S(0)=on) * PRODUCT_{t=0}^{998} A(on, on) = Pi(on) * A(on,on)^999

**Calculation**:
P = 0.5 * 0.8^999

0.8^999 is an incredibly small number (≈ 10^(-97)), but it is **NOT zero**. Since A(on,on) = 0.8 > 0, any power of it is still positive.

**Key principle**: A product of positive numbers is always positive. Only if A(on,on) = 0 would the probability be zero.

**Compare WS24/25 Q31-35**: Similar reasoning about HMM probabilities. Zero probability only when a ZERO entry in the relevant matrix makes a path impossible.

---

### Question 29 -- CORRECT
> **"If any entry in Pi is zero, then there is at least one state that is unreachable for the state transition Markov process."** --> **FALSE**

**Theory**: Pi defines the INITIAL state distribution. The transition matrix A determines which states are reachable over time.

**Analysis**:
Even if Pi(si) = 0 (state si can't be the initial state), state si might be reachable from other states through transitions.

**Example**: Pi = [0, 1] (always start off). But A(off, on) = 0.5 > 0, so the robot can transition to "on" at t=1. The "on" state is perfectly reachable despite having zero initial probability.

**Principle**: Pi determines starting conditions. A determines long-term reachability. A state is unreachable only if no state with positive probability has a path of positive-probability transitions leading to it.

---

### Question 30 -- CORRECT
> **"If any entry in the A matrix is zero, then there is at least one state that is unreachable."** --> **FALSE**

**Theory**: A(i,j) = 0 means state j can't be reached from state i in ONE step. But j might be reachable from i through intermediate states, or from other states entirely.

**Analysis**:
**Counterexample** with 3 states:
```
A = [0.5  0.5  0.0]    State 1 can go to 1 or 2, never directly to 3
    [0.0  0.5  0.5]    State 2 can go to 2 or 3, never directly to 1
    [0.5  0.0  0.5]    State 3 can go to 1 or 3, never directly to 2
```

A has zero entries, but every state is reachable: 1->2->3->1. The chain is irreducible despite having zeros in A.

In our vacuum robot: A(on,on)=0.8, A(on,off)=0.2, A(off,on)=0.5, A(off,off)=0.5. No zeros in A, but even if there were (e.g., A(on,off)=0), the "off" state might still be reachable from other paths.

---

### Question 31 -- CORRECT
> **"If o(1) = none, the robot cannot be on at t=1."** --> **FALSE**

**Theory**: P(S(1)=on | O(1)=none) depends on the emission probability B(on, none).

**Analysis**:
B(on, none) = 0.1 > 0. The robot CAN be on and produce "none" -- it's just unlikely (10% chance when on).

P(S(1)=on | O(1)=none) is proportional to P(O(1)=none | S(1)=on) * P(S(1)=on) = 0.1 * P(S(1)=on)

This is small but NOT zero. The robot can be on and happen to be quiet.

**Contrast**: If B(on, none) = 0, THEN observing none would rule out "on". But B(on, none) = 0.1 > 0.

---

### Question 32 -- CORRECT
> **"When o(t) = none, P(S(t) | o(1:t)) = [1, 0] regardless of o(1:t-1)."** --> **FALSE**

**Theory**: [1, 0] means P(S(t)=on) = 1 and P(S(t)=off) = 0. This would mean the robot is DEFINITELY on.

**Analysis**:
This claim is wrong for multiple reasons:

1. **B(off, none) = 1.0**: When off, the robot ALWAYS produces "none". So observing "none" is perfectly consistent with being off -- in fact, "none" is the ONLY thing the off state produces!

2. **Filtering uses prior history**: P(S(t) | o(1:t)) depends on the entire observation history. If previous observations were all "none", the robot is likely off (since off always produces "none"). The filtered distribution would be heavily weighted toward off, not on.

3. **Correct intuition**: Observing "none" INCREASES the probability of off (since B(off,none)=1.0 >> B(on,none)=0.1). The claim says the opposite.

**What would be closer to truth**: P(S(t) | o(t)=none) should be weighted toward [something, high_value] (toward off), not [1,0] (toward on).

---

### Question 33 -- CORRECT
> **"If we added another state 'charging', the B matrix would become a 3x3 matrix."** --> **TRUE**

**Theory**: B is the emission matrix with dimensions |S| x |O| (states x observations).

**Calculation**:
- Current: |S|=2 (on, off), |O|=3 (squeak, rattle, none) -> B is 2x3
- After adding state "charging": |S|=3, |O|=3 (unchanged) -> B is **3x3**

Adding a state adds a ROW to B (a new emission distribution for the new state).
Adding an observation would add a COLUMN to B.
States and observations are independent dimensions.

**Compare WS24/25 Q32-35**: Identical type of question. A is always |S|x|S| (square), B is |S|x|O|. Adding a state: A grows from NxN to (N+1)x(N+1), B grows from NxM to (N+1)xM.

---

## SECTION 6: Kalman Filters (Questions 34--38)

### The Model: Consumer Confidence Index

Simple 1-D Kalman Filter:
- **State**: X(t) = consumer confidence (real-valued, scalar)
- **Observation**: Z(t) = survey outcome (real-valued, scalar)

**Model equations**:
- Transition: X(t+1) = A * X(t) + epsilon_x, epsilon_x ~ N(0, Sigma_x)
- Observation: Z(t) = B * X(t) + epsilon_z, epsilon_z ~ N(0, Sigma_z)

Since everything is 1-D (scalar):
- A is a scalar (how confidence evolves)
- B is a scalar (how survey relates to confidence)
- Sigma_x is a scalar (process noise variance)
- Sigma_z is a scalar (observation noise variance)

---

### Question 34 -- CORRECT
> **"With this type of model, we cannot model a survey technique whose precision changes with the actual level of confidence."** --> **TRUE**

**Theory**: In a standard Kalman Filter, Sigma_z (observation noise) is a FIXED parameter. It does NOT depend on the state X(t).

**Analysis**:
"Survey precision changes with confidence level" means:
- When confidence is high, survey might be more/less precise
- This requires Sigma_z to be a FUNCTION of X(t): Sigma_z(X(t))

But in the Kalman Filter model, Sigma_z is a constant. The noise Z(t) - B*X(t) ~ N(0, Sigma_z) is always the same distribution regardless of X(t).

To model state-dependent noise, you'd need a **nonlinear** or **heteroscedastic** model, not a standard Kalman filter.

**Compare WS24/25 Q37**: "Cannot model GPS precision that changes with speed" -- identical reasoning. Sigma_z is fixed in standard KF.

---

### Question 35 -- CORRECT
> **"With this model, we cannot model the fact that consumer confidence tends to worsen more quickly than it improves."** --> **TRUE**

**Theory**: The Kalman Filter transition model is LINEAR: X(t+1) = A * X(t) + noise.

**Analysis**:
"Worsen more quickly than improve" means the dynamics are **asymmetric** -- the rate of change depends on the direction of change. This requires a nonlinear transition function:
- If X is decreasing: X(t+1) = f_decrease(X(t))
- If X is increasing: X(t+1) = f_increase(X(t))

A linear model X(t+1) = A * X(t) + noise is inherently **symmetric**. The coefficient A applies equally regardless of whether X is increasing or decreasing. The additive Gaussian noise is also symmetric.

**Principle**: Kalman Filters assume linearity. Any asymmetric, direction-dependent, or state-dependent dynamics violate this assumption.

---

### Question 36 -- CORRECT
> **"The coefficient matrix A can be used to encode certain assumptions about the expected precision of the survey."** --> **FALSE**

**Theory**: Each parameter in the Kalman Filter has a specific role:
- **A**: State transition dynamics (how confidence evolves over time)
- **B**: Observation mapping (how survey relates to true confidence)
- **Sigma_x**: Process noise (uncertainty in state transitions)
- **Sigma_z**: Observation noise (**precision of the survey**)

**Analysis**:
A encodes how the STATE evolves. Survey precision is about the OBSERVATION model, encoded in Sigma_z (noise variance) and B (scaling).

A has nothing to do with survey precision. If you want to model a more precise survey, you decrease Sigma_z. If you want to change how the survey maps to confidence, you change B.

**Compare WS24/25 Q39**: "A encodes assumptions about GPS precision" -- FALSE, same reasoning. A is about state dynamics, not observation precision.

---

### Question 37 -- CORRECT
> **"If the consumer confidence improves over a long time, the A matrix will change."** --> **FALSE**

**Theory**: The Kalman Filter makes a **stationarity assumption**: all model parameters (A, B, Sigma_x, Sigma_z) are CONSTANT over time.

**Analysis**:
The STATE X(t) changes over time -- that's the whole point of the model. But the PARAMETERS that govern how the state evolves are fixed.

If confidence improves, X(t) increases. But A (the transition coefficient), Sigma_x (process noise), etc. remain the same. The model is like "the rules of the game are fixed, but the game state changes."

**Compare WS24/25 Q41**: "If bicycle speeds up, A matrix changes" -- FALSE, identical reasoning.

**Important distinction**:
- X(t) changes: YES (that's what we're tracking)
- A changes: NO (stationarity assumption)
- Sigma changes: NO (fixed parameters)

---

### Question 38 -- CORRECT
> **"To adapt this model to a population with a more volatile mood (stronger tendencies to change, still symmetric), adapt: ___"** --> **Sigma_x**

**Theory**: "Volatile mood" means the state changes more rapidly and unpredictably between time steps. This is captured by the PROCESS NOISE.

**Analysis of each option**:

| Parameter | Controls | Would changing it model volatility? |
|-----------|----------|-------------------------------------|
| **A** | Deterministic trend in state evolution | No -- A controls the systematic part (e.g., "confidence tends to persist"). Changing A would change the expected value, not the variability |
| **B** | How observation maps to state | No -- this is about the survey, not about mood changes |
| **Sigma_x** | **Process noise** -- randomness in state transitions | **YES!** Larger Sigma_x means more unpredictable state changes between time steps = more volatile mood |
| **Sigma_z** | Observation noise -- survey precision | No -- this is about measurement quality, not actual mood volatility |

**Sigma_x** is the correct answer. A more volatile population has larger process noise -- their confidence jumps around more unpredictably from one time step to the next.

**Intuition**: 
- Sigma_x = 0: Confidence follows A*X perfectly (no randomness in transitions)
- Sigma_x small: Confidence mostly follows the trend with small fluctuations
- Sigma_x large: Confidence is highly volatile, jumping around unpredictably

---

## SUMMARY OF MISTAKES

### Q2: "Full joint has only 9 non-redundant parameters" --> TRUE
**Lesson**: Count BN parameters by summing CPT sizes: P(W)=1 + P(M|W)=2 + P(F|W)=2 + P(R|M,F)=4 = 9. The "full joint" in a BN context means the joint as parameterised by the model's CPTs, not the 2^n-1 parameters of an unrestricted joint table.

### Q13: "P(F|not_w) needs not consider R" --> TRUE
**Lesson**: When a variable is observed as evidence, it "screens off" everything on the other side. F's only parent is W. Given W (observed), F is conditionally independent of everything else in the network (including R). You just look up P(F|W=false) in the CPT.

---

## CROSS-EXAM COMPARISON TABLE

| Topic | WS25/26 Qs | WS24/25 Qs | WS23/24 | Retake 24 |
|-------|-----------|-----------|---------|-----------|
| Independence / d-separation | Q1,3,7 | Q1-8 | ✓ | ✓ |
| Parameter counting | Q2 | Q9,24,26 | ✓ | ✓ |
| Edge reversal / structure | Q4,6 | Q11 | ✓ | |
| BN factorisation | Q5 | Q5 | | ✓ |
| No-edges model | Q8-10 | | | |
| Variable elimination | Q11 | Q11 | ✓ | ✓ |
| Evidential reasoning | Q12 | Q12 | | |
| Conditional independence in queries | Q13,14 | Q13 | ✓ | |
| Forward sampling | Q15 | | | ✓ |
| Gibbs / Markov blanket | Q16 | Q17 | ✓ | ✓ |
| Rejection sampling | Q17 | Q15-16 | | |
| Likelihood function properties | Q19,21,22 | Q18-21 | | |
| Laplace smoothing | Q20 | Q20 | ✓ | |
| Bayesian estimation / posterior | Q18,23 | Q22-23 | ✓ | ✓ |
| BIC / structure learning | Q24-27 | Q27-30 | | ✓ |
| HMM matrix sizes | Q33 | Q32-35 | ✓ | ✓ |
| HMM reachability | Q28-30 | | | |
| HMM filtering | Q31-32 | Q31,36 | ✓ | ✓ |
| KF: what each parameter controls | Q34-38 | Q37-41 | | ✓ |
