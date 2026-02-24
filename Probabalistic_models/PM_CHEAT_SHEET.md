# Probabilistic Models -- Comprehensive Cheat Sheet

---

## 1. FUNDAMENTAL CONCEPTS & NOTATION

### Random Variables (RVs)
- **Boolean**: `Weather = {sunny, rainy}` -- 2 values
- **Discrete**: `Weather = {sunny, rainy, cloudy, snow}` -- finite set
- **Continuous**: `Temp = 22.6` -- real-valued

### Events
- **Atomic event** = complete assignment of values to ALL variables in the world (one full row of the joint distribution table). Atomic events are **mutually exclusive** and **exhaustive**.
- **General event** = a logical formula (proposition) that can be true in multiple atomic events. E.g., `Cavity = true`.

### Probability Basics
| Concept | Formula |
|---------|---------|
| Axioms | 0 <= P(a) <= 1; P(true) = 1; P(false) = 0; P(a v b) = P(a) + P(b) - P(a ^ b) |
| Full joint distribution | P(X1, X2, ..., Xn) -- specifies probability for every atomic event |
| Marginalisation | P(X) = SUM_y P(X, Y=y) |
| Conditioning | P(X \| Y) = P(X, Y) / P(Y) |
| Product rule | P(X, Y) = P(X \| Y) * P(Y) = P(Y \| X) * P(X) |
| Chain rule | P(X1,...,Xn) = P(X1) * P(X2\|X1) * P(X3\|X1,X2) * ... |
| **Bayes' rule** | **P(H \| e) = P(e \| H) * P(H) / P(e)** |
| Law of total probability | P(X) = SUM_y P(X \| Y=y) * P(Y=y) |
| Normalisation trick | P(X \| e) = alpha * P(X, e) where alpha = 1/P(e) |

### Independence
- **Marginal independence**: X _|_ Y iff P(X,Y) = P(X)*P(Y) iff P(X|Y) = P(X)
- **Conditional independence**: (X _|_ Y | Z) iff P(X,Y|Z) = P(X|Z)*P(Y|Z)
- Conditional independence does NOT imply marginal independence and vice versa!

### EXAM TRAPS:
- P(a | b, c) = P(a | c) does NOT mean a is independent of b -- it means a is conditionally independent of b GIVEN c
- "The full joint distribution contains all information needed" -- TRUE, you can always marginalise/condition to get any query
- Number of independent parameters in a full joint over n boolean vars = **2^n - 1** (the last one is determined by normalisation)

---

## 2. BAYESIAN NETWORKS (BNs) -- REPRESENTATION

### What is a BN?
A BN is a **directed acyclic graph (DAG)** where:
- Each node = a random variable
- Each edge = a direct dependency
- Each node has a **conditional probability table (CPT)**: P(Xi | Parents(Xi))

### Chain Rule for BNs
P(X1,...,Xn) = PRODUCT_i P(Xi | Parents(Xi))

This is always valid in a BN -- it's a **compact factored representation** of the full joint.

### Topological Ordering
- Variables must be orderable such that parents come before children
- DAG guarantees at least one such ordering exists
- Multiple valid orderings may exist

### Compactness / Number of Parameters
For a BN with boolean variables:
- A variable with k boolean parents needs **2^k** rows in its CPT
- Independent parameters per variable with k parents and d values: **(d-1) * d_parent1 * d_parent2 * ...**
- For boolean: each variable with k parents has **2^k** independent parameters

**EXAM Q (WS24/25 Q9)**: For the car diagnosis network with specific structure, count parameters carefully by summing CPT sizes for each node.

**EXAM Q (WS24/25 Q24)**: "How many unconditional distributions have to be estimated?" = number of root nodes (nodes with no parents). Answer: 4.
**EXAM Q (WS24/25 Q26)**: "How many independent parameters for variable I with parents D, S, G, L (all boolean except G with 3 values)?" = (2-1) * 2 * 2 * 3 * 2 = **24**

### Independence in BNs (D-separation)

Three fundamental patterns:

1. **Chain**: A -> B -> C
   - A and C are dependent (marginally)
   - A _|_ C | B (conditionally independent given B)

2. **Common cause (fork)**: A <- B -> C
   - A and C are dependent (marginally)
   - A _|_ C | B (conditionally independent given B)

3. **Common effect (v-structure / collider)**: A -> B <- C
   - A _|_ C (marginally independent!)
   - A and C become DEPENDENT given B (or any descendant of B) -- **explaining away**

### D-Separation Algorithm
X _|_ Y | Z iff every path from X to Y is "blocked" by Z:
- A path is blocked if it contains a node M such that:
  - M is in Z and the connection is chain or fork, OR
  - M is a collider (v-structure) and neither M nor any descendant of M is in Z

### EXAM TRAPS (from WS24/25 Q1-Q8):
- **Q1**: P(Starter_broken | Car_won't_start) = P(No_oil | Car_won't_start) is **FALSE** -- different CPTs, different parents
- **Q5**: (Starter_broken _|_ Battery_dead) is **TRUE** if there's no active path between them (check d-separation in the original graph)
- **Q6**: P(Battery_age) = P(Battery_age | Battery_meter) is **FALSE** -- Battery_meter is a descendant, so they're generally dependent
- **Q7**: P(Battery_flat | Lights, Battery_dead) = P(Battery_flat | Battery_dead) is **FALSE** -- Lights may provide info about Battery_flat through a shared ancestor/descendant structure
- **Q8**: (No_charging _|_ No_oil | Alternator_broken, Fanbelt_broken) is **TRUE** -- given all intermediate causes, the two are separated
- **Q11**: Adding edge from Car_won't_start to Battery_meter would be illegal is **FALSE** -- it's legal as long as no cycle is created
- **Q12**: "No topological ordering with all orange variables first" can be **TRUE** if some orange variable has a non-orange parent

### EXAM Q: "Can a model with this structure fit all possible joint distributions?" -- Only if the structure doesn't impose independence constraints that exclude certain distributions. Generally **FALSE** unless the graph is fully connected.

---

## 3. EXACT INFERENCE

### Inference by Enumeration
P(X | e) = alpha * SUM_{hidden} P(X, e, hidden)

Compute by summing over all assignments to hidden variables. Works but **exponential** in number of hidden variables.

### Variable Elimination (VE)
Key idea: **push sums inward** (distribute sum over product) to avoid computing full joint.

Steps:
1. Write query as sum-of-products from BN factorisation
2. Choose an elimination ordering for hidden variables
3. For each hidden variable (innermost sum first):
   - Collect all factors that mention this variable
   - Multiply them together
   - Sum out the variable -> produces a new (smaller) factor
4. Multiply remaining factors and normalise

**Factors**: A factor f(X1,...,Xk) is a table mapping variable assignments to real numbers.
- **Pointwise product**: f1(X,Y) * f2(Y,Z) = f3(X,Y,Z)
- **Summing out**: SUM_Y f(X,Y) = g(X)

### Complexity
- VE is polynomial in treewidth of the BN
- Worst case still **NP-hard** (general BN inference is NP-hard)
- Elimination ordering matters hugely for efficiency
- Finding optimal ordering is itself NP-hard

### EXAM TRAPS:
- "We cannot calculate P(Battery_flat) without knowing Battery_dead etc." -- **FALSE**: we can always marginalise over unknown variables
- "The dipstick tells us nothing about the oil light" -- **FALSE**: observing dipstick gives info about Oil_ok which affects oil light
- "P(Car_won't_start) gives probability when all green variables known" -- **FALSE**: P(Car_won't_start) is a prior, not conditioned on evidence

---

## 4. APPROXIMATE INFERENCE

### Forward (Prior) Sampling
- Sample variables in topological order
- Each variable sampled from P(Xi | parents(Xi)) using already-sampled parent values
- Samples approximate the **prior** distribution P(X1,...,Xn)
- Consistent: converges to true distribution as N -> infinity

### Rejection Sampling
- Generate forward samples
- **Reject** (discard) any sample inconsistent with evidence
- Keep only samples matching evidence
- Problem: if evidence is unlikely, almost all samples rejected -> very inefficient

### Likelihood Weighting
- Fix evidence variables to their observed values (don't sample them)
- Sample non-evidence variables in topological order
- **Weight** each sample by product of P(ei | parents(ei)) for each evidence variable
- Much more efficient than rejection -- no samples wasted

**Key insight**: Evidence variables are NOT sampled; they are clamped. Non-evidence variables above (ancestors of) evidence in the ordering are sampled from their prior -> they don't get "steered" by evidence -> samples can still be poor if evidence is deep in the network.

### EXAM TRAPS (WS24/25 Q14-17):
- **Q14**: "Given Car_won't_start as evidence, all orange variables contribute weight 1.0 in likelihood weighting" -- **TRUE** because orange variables (diagnoses) are NOT evidence, so they are sampled normally (weight contribution = 1.0 since they're not clamped)
- **Q16**: "Given Car_won't_start as evidence, all green variables contribute weight 1.0" -- **TRUE** because green variables (tests) are also NOT evidence in this query (only Car_won't_start is evidence)
- **Q15**: "In query with all green variables as evidence, 9 variables sampled in forward sampling" -- Check: you sample ALL non-evidence variables. Count total variables minus evidence variables.
- **Q17**: "No variable has Markov blanket > 5 variables" -- **FALSE**: Markov blanket = parents + children + co-parents. Can be large.

### Markov Chain Monte Carlo (MCMC)
Instead of independent samples, generate a **chain** of samples where each depends on the previous.

**Gibbs Sampling** (special case of MCMC):
1. Start with arbitrary assignment to all non-evidence variables
2. Repeat: pick a non-evidence variable Xi, resample it from:
   P(Xi | markov_blanket(Xi))
3. The chain converges to the true posterior distribution

**Markov Blanket** of Xi = {parents(Xi)} ∪ {children(Xi)} ∪ {other parents of children(Xi)}

**Metropolis-Hastings** (general MCMC):
1. Current state x, propose new state x' from proposal distribution q(x'|x)
2. Accept with probability: min(1, [P(x')*q(x|x')] / [P(x)*q(x'|x)])
3. If accepted, move to x'; otherwise stay at x

Properties:
- **Burn-in**: initial samples may not represent target distribution; discard them
- **Mixing time**: how long until chain is approximately sampling from target
- **Detailed balance**: sufficient condition for convergence: P(x)*T(x->x') = P(x')*T(x'->x)

---

## 5. PARAMETER LEARNING

### Setup
Given: BN structure (graph), dataset D of N complete observations.
Goal: Learn the CPT parameters θ.

### Maximum Likelihood Estimation (MLE)
- Choose θ that maximizes P(D | θ)
- Equivalently, maximise **log-likelihood**: LL(θ) = SUM_i log P(di | θ)

**Key property of BNs**: Log-likelihood **decomposes** by variable:
LL(θ) = SUM_i SUM_j LL(θ_Xi | parents)

So we can learn each CPT independently!

**MLE for discrete variables**:
θ_MLE(Xi=x | Parents=pa) = Count(Xi=x, Parents=pa) / Count(Parents=pa)

Just count frequencies in the data!

### Thumbtack / Coin example (Bernoulli)
- N flips, k heads, (N-k) tails
- Likelihood: L(θ) = θ^k * (1-θ)^(N-k)
- MLE: θ_hat = k/N

### EXAM TRAPS (WS24/25 Q18-21):
- **Q18**: "10x more data with same ratio -> likelihood keeps max at same place but becomes flatter" -- **FALSE**: max stays at same θ=k/N (since ratio is same), but the curve becomes **narrower/sharper** (more data = more certainty)
- **Q19**: "Function value at max would be 10x as high" -- **FALSE**: likelihood is a product of more terms, so value at max actually gets much smaller
- **Q20**: "Likelihood function would range from 0 to 10 on x-axis" -- **FALSE**: θ always ranges from 0 to 1 (it's a probability)
- **Q21**: "Maximum shifts toward 0.5" -- **FALSE**: MLE = k/N = 30/50 = 0.6, same as 3/5

### Problem: Data Fragmentation
With many parents, CPT has many entries but few data points per entry -> unreliable estimates.
Extreme case: 0 counts -> θ = 0 -> assigns probability 0 to unseen events!

### Laplace Correction (Smoothing)
Add "virtual counts" (pseudocounts) to each cell:
θ(Xi=x | pa) = [Count(Xi=x, pa) + α] / [Count(pa) + α*|Xi|]

With α=1 (add-one smoothing / Laplace correction).

### Bayesian Parameter Estimation
Instead of single point estimate, maintain a **distribution over θ**: P(θ | D)

**Prior**: P(θ) -- our belief before seeing data
**Posterior**: P(θ | D) = P(D | θ) * P(θ) / P(D) (Bayes' rule!)

**Conjugate Prior**: A prior that, when combined with the likelihood, produces a posterior of the same family.
- Bernoulli likelihood + **Beta prior** -> Beta posterior
- Multinomial likelihood + **Dirichlet prior** -> Dirichlet posterior
- Gaussian likelihood + **Gaussian prior** (for mean) -> Gaussian posterior

### Beta Distribution
Beta(θ | a, b) ∝ θ^(a-1) * (1-θ)^(b-1)

- Prior: Beta(a, b) -- hyperparameters a, b encode "virtual observations"
- After seeing k heads, (N-k) tails: Posterior = Beta(a+k, b+N-k)
- **MAP estimate**: θ_MAP = (a+k-1) / (a+b+N-2)
- **Posterior mean**: θ_PM = (a+k) / (a+b+N)

### MAP vs MLE
- MAP = MLE + prior influence
- With symmetric prior at 0.5 (e.g., Beta(2,2)) and data showing θ_MLE = 0.6:
  - **θ_MAP < θ_MLE** (prior pulls toward 0.5)
- With more data, MAP -> MLE (data overwhelms prior)

### EXAM TRAPS (WS24/25 Q22-23):
- **Q22**: "With prior at 0.5, MAP estimate is __ than MLE (0.6)" -- **smaller than** θ_MLE (prior pulls toward 0.5, which is less than 0.6)
- **Q23**: "With 10x more data, MAP estimate will..." -- **move closer to θ_MLE** (more data dominates the prior)

### MLE for Gaussian
Given data points x1,...,xN from N(μ, σ²):
- μ_MLE = (1/N) * SUM xi = sample mean
- σ²_MLE = (1/N) * SUM (xi - μ_MLE)² = sample variance (biased)

For multivariate Gaussian N(μ, Σ):
- μ_MLE = sample mean vector
- Σ_MLE = sample covariance matrix

---

## 6. STRUCTURE LEARNING & MODEL SELECTION

### Model Dimension
The number of independent parameters in a BN.

### Likelihood and Structure
- Adding edges can only **increase or maintain** the likelihood (never decrease)
- More edges = more parameters = better fit to training data
- But: risk of **overfitting**!

### BIC Score (Bayesian Information Criterion)
BIC = LL(θ_MLE; D) - (d/2) * log(N)

Where:
- LL = log-likelihood with ML parameters
- d = number of independent parameters (model dimension)
- N = number of data points

BIC penalises model complexity. A model with higher BIC is preferred.

### EXAM TRAPS (WS24/25 Q25-30):
- **Q25**: "Can this structure fit all possible joint distributions?" -- **FALSE**: BN structure imposes independence constraints
- **Q27**: "Adding edges -> BIC can only decrease" -- **FALSE**: BIC can increase (if likelihood gain outweighs complexity penalty) or decrease (if complexity penalty dominates)
- **Q28**: "If likelihood increases with new edge, BIC also increases" -- **FALSE**: the complexity penalty might outweigh the likelihood gain
- **Q29**: "ML estimates for variables D, G, L won't change when adding edge D->S" -- **TRUE**: because their CPTs don't change (only S's CPT changes since S gets a new parent)
- **Q30**: "Likelihood of new model cannot be lower" -- **TRUE**: adding edges can only increase or maintain likelihood

---

## 7. GAUSSIAN MIXTURE MODELS (GMMs) & EM ALGORITHM

### GMM
P(x) = SUM_k w_k * N(x | μ_k, Σ_k)

- K mixture components, each a Gaussian N(μ_k, Σ_k)
- w_k = mixing weights (sum to 1)
- Parameters: {w_k, μ_k, Σ_k} for k=1,...,K

### EM Algorithm (Expectation-Maximisation)
Used when there are **latent/hidden variables** (like cluster assignments).

1. **E-step**: Compute expected values of hidden variables given current parameters
   - For GMM: compute "responsibilities" r_ik = P(component k | data point i)
   - r_ik = w_k * N(xi | μ_k, Σ_k) / SUM_j w_j * N(xi | μ_j, Σ_j)

2. **M-step**: Update parameters using the expected hidden variable values
   - μ_k = SUM_i r_ik * xi / SUM_i r_ik
   - Σ_k = SUM_i r_ik * (xi - μ_k)(xi - μ_k)^T / SUM_i r_ik
   - w_k = (1/N) * SUM_i r_ik

Properties:
- Guaranteed to **increase** (or maintain) the log-likelihood at each step
- Converges to a **local** maximum (not necessarily global)
- Sensitive to initialisation
- Is a general framework -- applies to any model with latent variables

---

## 8. DYNAMIC BAYESIAN NETWORKS (DBNs)

### Temporal Models
Model how the world evolves over time: X(0), X(1), X(2), ...

### Simplifying Assumptions
1. **Discrete time** slices: t = 0, 1, 2, ...
2. **Markov assumption (1st order)**: X(t+1) depends only on X(t), not on X(t-1), X(t-2), ...
   P(X(t+1) | X(0:t)) = P(X(t+1) | X(t))
3. **Stationarity**: Transition model P(X(t+1) | X(t)) is the same for all t

### State-Observation Model
- **State variables** S(t): hidden, evolve over time
- **Observation variables** O(t): observed, depend on current state
- **Transition model**: P(S(t+1) | S(t))
- **Observation model**: P(O(t) | S(t))
- **Initial state**: P(S(0)) = Π

### Inference Tasks in Temporal Models

| Task | Query | Description |
|------|-------|-------------|
| **Filtering** | P(S(t) \| O(0:t)) | Current state given all observations so far |
| **Prediction** | P(S(t+k) \| O(0:t)), k>0 | Future state given current observations |
| **Smoothing** | P(S(k) \| O(0:t)), k<t | Past state given all observations (including future ones) |
| **Decoding** | argmax_{S(0:t)} P(S(0:t) \| O(0:t)) | Most likely state SEQUENCE |
| **Likelihood** | P(O(0:t)) | Probability of observation sequence |

### Filtering (Forward Algorithm)
Recursive update:
1. **Predict**: P(S(t+1) | o(0:t)) = SUM_s P(S(t+1) | S(t)=s) * P(S(t)=s | o(0:t))
2. **Update**: P(S(t+1) | o(0:t+1)) = alpha * P(o(t+1) | S(t+1)) * P(S(t+1) | o(0:t))

where alpha is a normalisation constant.

### Smoothing (Forward-Backward Algorithm)
P(S(k) | o(0:t)) = alpha * f(k) * b(k)
- f(k) = forward message = P(S(k), o(0:k)) [from filtering]
- b(k) = backward message = P(o(k+1:t) | S(k)) [computed backwards from t]

### EXAM TRAP (WS24/25 Q36):
- "Smoothed distribution P(S(0) | O) always equals Π" -- **FALSE**: smoothing uses ALL observations (past and future), so it updates the initial distribution. Only if observations provide no information would it stay at Π.

---

## 9. HIDDEN MARKOV MODELS (HMMs)

### Formal Specification
An HMM is a state-observation model where:
- **States** S = {s1, ..., sN} are **discrete** and **hidden**
- **Observations** O = {o1, ..., oM} are **discrete**

Parameters (λ = {Π, A, B}):
- **Π** = initial state distribution: Π_i = P(S(0) = si), vector of size N
- **A** = state transition matrix: A_ij = P(S(t+1) = sj | S(t) = si), matrix N×N
- **B** = observation/emission matrix: B_ij = P(O(t) = oj | S(t) = si), matrix N×M

### EXAM TRAPS (WS24/25 Q31-35) -- Matrix Sizes:
- **A is always N×N (square!)** -- transitions between N states
- **B is always N×M** (N states × M observations)
- Adding an observation value: B becomes N×(M+1), A stays N×N
- Adding a state: A becomes (N+1)×(N+1), B becomes (N+1)×M, Π becomes (N+1)
- States and observations are **independent dimensions** -- adding a state does NOT require adding an observation

### "Most probable observation at time 0"
Calculate P(O(0) = oj) = SUM_i Π_i * B_ij for each observation oj, pick the maximum.

**Example** (WS24/25 Q31):
- S = {rainy, cloudy, sunny}, O = {dry, medium, humid}
- Π = [0.2, 0.3, 0.5]
- B = [[0.1, 0.3, 0.6], [0.5, 0.4, 0.2], [0.4, 0.3, 0.2]]
  (rows = states, columns = observations)

P(O(0) = dry) = 0.2*0.1 + 0.3*0.5 + 0.5*0.4 = 0.02 + 0.15 + 0.20 = 0.37
P(O(0) = medium) = 0.2*0.3 + 0.3*0.4 + 0.5*0.3 = 0.06 + 0.12 + 0.15 = 0.33
P(O(0) = humid) = 0.2*0.6 + 0.3*0.2 + 0.5*0.2 = 0.12 + 0.06 + 0.10 = 0.28

Most probable: **dry** (0.37), not humid. -> Q31 answer: FALSE.

### HMM Inference Algorithms

#### Filtering (Forward Algorithm)
α(t) = vector of forward variables: α_i(t) = P(S(t)=si, O(0:t)=o(0:t))

Recursion:
- α_i(0) = Π_i * B(i, o(0))
- α_i(t+1) = B(i, o(t+1)) * SUM_j α_j(t) * A(j,i)

In matrix form: α(t+1) = B_diag(o(t+1)) * A^T * α(t)

P(S(t) | o(0:t)) = normalise(α(t))

#### Prediction
P(S(t+k) | o(0:t)) = A^k * P(S(t) | o(0:t))

As k -> infinity, prediction converges to the **stationary distribution** of the Markov chain.

#### Smoothing (Forward-Backward)
β_i(t) = P(O(t+1:T) | S(t)=si) -- backward variable

Recursion (backwards from T):
- β_i(T) = 1
- β_i(t) = SUM_j A(i,j) * B(j, o(t+1)) * β_j(t+1)

Smoothed: P(S(t) | o(0:T)) = normalise(α(t) ⊙ β(t)) (element-wise multiply)

#### Viterbi Algorithm (Decoding)
Find the most likely STATE SEQUENCE (not just most likely state at each time).

Like forward algorithm but replace SUM with MAX:
- δ_i(t+1) = B(i, o(t+1)) * MAX_j [δ_j(t) * A(j,i)]
- Keep backpointers to trace back the best path

#### Likelihood Computation
P(O(0:T)) = SUM_i α_i(T)

### Learning HMMs: Baum-Welch Algorithm
An instance of EM:
- **E-step**: Compute expected state occupancies and transitions using forward-backward
  - γ_i(t) = P(S(t)=si | O, λ) -- state occupancy
  - ξ_ij(t) = P(S(t)=si, S(t+1)=sj | O, λ) -- transition occupancy
- **M-step**: Re-estimate parameters:
  - Π_i = γ_i(0)
  - A_ij = SUM_t ξ_ij(t) / SUM_t γ_i(t)
  - B_ij = SUM_{t: O(t)=oj} γ_i(t) / SUM_t γ_i(t)

---

## 10. KALMAN FILTERS

### When to Use
- **Continuous** state variables (not discrete)
- **Linear** dependencies between time steps
- **Gaussian** noise

### Formal Specification
State-Observation model with:
- State: **x(t)** ∈ R^n (continuous vector)
- Observation: **z(t)** ∈ R^m (continuous vector)

**Transition model**: x(t+1) = **A** * x(t) + ε_x, where ε_x ~ N(0, **Σ_x**)
**Observation model**: z(t) = **B** * x(t) + ε_z, where ε_z ~ N(0, **Σ_z**)
**Initial state**: x(0) ~ N(**μ_0**, **Σ_0**)

Parameters:
- **A** = state transition matrix (n×n) -- how state evolves
- **B** = observation matrix (m×n) -- how observations relate to state
- **Σ_x** = process noise covariance (n×n)
- **Σ_z** = observation noise covariance (m×m)

### Bicycle Example (from lecture/exam)
State: x(t) = [P(t), V(t)]^T (position, velocity)
Observation: z(t) = GPS reading

A = [[1, Δt], [0, 1]] (position += velocity*Δt, velocity stays ~same)
B = [1, 0] (GPS observes position only)

### Tracking (Filtering) Algorithm
At each time step:
1. **Prediction** (time update):
   - μ_pred = A * μ(t)
   - Σ_pred = A * Σ(t) * A^T + Σ_x

2. **Update** (measurement update):
   - K = Σ_pred * B^T * (B * Σ_pred * B^T + Σ_z)^(-1)  [**Kalman gain**]
   - μ(t+1) = μ_pred + K * (z(t+1) - B * μ_pred)
   - Σ(t+1) = (I - K * B) * Σ_pred

**Key property**: If initial state is Gaussian and all models are linear Gaussian, then the filtered state is **always Gaussian** -> only need to track μ and Σ (no approximation needed!).

### EXAM TRAPS (WS24/25 Q37-41):
- **Q37**: "Cannot model GPS precision that changes with speed" -- **TRUE**: in a standard Kalman filter, Σ_z is constant (doesn't depend on state). Would need a nonlinear extension.
- **Q38**: "If V(t+1) also depended on P(t), A matrix needs additional column" -- **FALSE**: A is n×n where n = number of state variables. Adding a dependency P(t)->V(t+1) just changes a value IN the A matrix (A[2,1] would become nonzero), it doesn't change the matrix dimensions.
- **Q39**: "Coefficient matrix A encodes assumptions about GPS sensor precision" -- **FALSE**: A is the state transition matrix. GPS precision is encoded in Σ_z (and B determines which states are observed).
- **Q40**: "If V(t+1) depends on P(t), A matrix size doubles" -- **FALSE**: matrix stays n×n, only entries change.
- **Q41**: "If bicycle speeds up, A matrix changes" -- **FALSE**: A is a fixed parameter (stationarity assumption). The state values change, not the model parameters.

### Limitations of Kalman Filters
- Only works for **linear** models with **Gaussian** noise
- Cannot handle multi-modal distributions
- For non-linear: Extended Kalman Filter (EKF), Unscented Kalman Filter (UKF)
- For discrete + continuous: **Switching Kalman Filter**

### Switching Kalman Filter
Combines HMM (discrete mode variable) with Kalman Filter (continuous state):
- Discrete "regime" variable determines which set of Kalman parameters to use
- E.g., driving modes: city driving vs. highway

---

## 11. EXAM QUESTION PATTERNS & STRATEGIES

### Question Type: True/False on Independence
Strategy:
1. Draw the BN graph
2. Identify the query: (X _|_ Y | Z)?
3. Apply d-separation: check all paths from X to Y
4. Remember: v-structures (colliders) are BLOCKED unless conditioned on (or descendant in evidence)

### Question Type: Counting Parameters
Strategy:
1. For each variable Xi, count |Parents(Xi)| and value domains
2. Independent params for Xi = (|Xi| - 1) * PRODUCT |Parent_j|
3. Sum over all variables

### Question Type: MLE / MAP / Bayesian Estimation
Strategy:
- MLE = counts/frequencies
- MAP includes prior: pushed toward prior mode
- More data -> MAP approaches MLE
- Likelihood becomes sharper (narrower) with more data, not flatter

### Question Type: HMM Matrix Operations
Strategy:
- A is ALWAYS square (N×N)
- B is N×M (states × observations)
- States and observations have independent cardinalities
- Adding state: A grows from N×N to (N+1)×(N+1), B from N×M to (N+1)×M
- Adding observation: B from N×M to N×(M+1), A unchanged

### Question Type: Kalman Filter Parameters
Strategy:
- A: state transition (n×n), encodes dynamics
- B: observation matrix (m×n), maps state to observation space
- Σ_x: process noise (n×n)
- Σ_z: observation noise (m×m)
- All are FIXED parameters (stationarity) -- don't change as state evolves
- Adding dependency between existing state variables: doesn't change matrix SIZE

### Question Type: Likelihood Weighting
Strategy:
- Evidence variables are CLAMPED (not sampled)
- Non-evidence variables ARE sampled normally
- Weight = product of P(ei | parents(ei)) for each evidence variable
- If a variable is NOT evidence, it contributes weight 1.0 (not multiplied into weight)
- Markov blanket(X) = parents + children + co-parents of children

---

## 12. KEY FORMULAS QUICK REFERENCE

| Formula | Expression |
|---------|-----------|
| Bayes' Rule | P(H\|e) = P(e\|H)P(H) / P(e) |
| BN Chain Rule | P(X1,...,Xn) = Π P(Xi\|Parents(Xi)) |
| MLE (discrete) | θ(x\|pa) = Count(x,pa) / Count(pa) |
| Laplace | θ(x\|pa) = (Count(x,pa)+1) / (Count(pa)+\|X\|) |
| Beta posterior | Beta(a+k, b+N-k) from prior Beta(a,b) + data (k heads, N-k tails) |
| MAP (Beta) | θ = (a+k-1)/(a+b+N-2) |
| Post. mean (Beta) | θ = (a+k)/(a+b+N) |
| BIC | LL - (d/2)log(N) |
| HMM forward | α_i(t+1) = B(i,o(t+1)) * Σ_j α_j(t)*A(j,i) |
| HMM backward | β_i(t) = Σ_j A(i,j)*B(j,o(t+1))*β_j(t+1) |
| Viterbi | δ_i(t+1) = B(i,o(t+1)) * max_j [δ_j(t)*A(j,i)] |
| Kalman predict | μ' = Aμ; Σ' = AΣA^T + Σ_x |
| Kalman update | K = Σ'B^T(BΣ'B^T + Σ_z)^{-1}; μ = μ'+K(z-Bμ'); Σ = (I-KB)Σ' |
| M-H acceptance | min(1, P(x')q(x\|x') / P(x)q(x'\|x)) |

---

## 13. PAST EXAM ANSWERS -- COMPLETE REFERENCE

### WS 2023/24 Exam Key Topics:
- BN structure and independencies
- Variable elimination
- Approximate inference (likelihood weighting, Gibbs sampling)
- Parameter learning (MLE, MAP)
- HMM inference

### WS 2024/25 Exam (37/41 = 90.24%):
All 41 questions covered above in respective sections. Mistakes were on:
- Q7: Battery_flat conditional independence (tricky d-separation)
- Q12: Topological ordering constraints
- Q14: Likelihood weighting -- which variables contribute to weight

### WS 2025/26 Exam Topics (based on PM_Exam_WS25_26.pdf):
- Conditional independence in BNs
- Number of parameters
- Variable elimination ordering
- Approximate inference methods
- Parameter learning with Beta priors
- HMM filtering/prediction
- Kalman filter matrices

### Retake Exam 2024 Key Topics:
- BN representation and semantics
- Independence assertions from graph structure
- Exact and approximate inference
- Parameter learning formulas
- Temporal models (HMM, Kalman)
