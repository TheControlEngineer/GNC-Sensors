# GNC-Sensors

##Lidar Design and Validation Ongoing


## Range Testing 

Initial four tests were conducted on the LiDAR sensor model. 

### Test 1  Ranging


Goal:- The goal is to confirm the deterministic measurement pipeline reproduces ground truth range with no stochastic terms.

Method:- The noise into the LiDAR measurements were deactivated, and the measured range was compared against true range.

Passing Criteria:-  Measured range equals true range at machine precision for every sampled distance from 0 to max range.

Results :- **Passed**

<img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/5f31926b-65fa-4dea-8ab3-126493ffc053" />

### Test 2 FOV Validity 

Goal:- Identify where measurement validity transitions between valid and out_of_fov relative to half FoV limits.

Method:- A target is placed at the edge of the field of view and swept across to the other side, the resultant measurements are checked to determine valid and invalid measurements.

Passing Criteria:- Sweep starts invalid, center is valid, sweep ends invalid, and epsilon checks at both boundaries match expected valid/invalid transitions.

Result:- **Passed**

<img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/777622d1-2660-47d6-b147-9dd3ae58273b" />

### Test 3 Range Validity
This test checks measurement validity at different ranges 

Goal:- Identify where measurement validity transitions between valid and out_of_range at range_min and range_max.

Method:-  we sweep target range from below minimum range to above maximum range and evaluate boundary behavior.

Passing Criteria:- Near range_min, minus epsilon is invalid while boundary and plus epsilon are valid; near range_max, minus epsilon and boundary are valid while plus epsilon is invalid; sweep starts invalid, is valid inside limits, and ends invalid.

Result:- **Passed**

<img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/fd216479-285c-4248-ab62-4ddfdc11acd9" />

### Test 4 Beam Geometry Validation

Goal:-  Confirm hit geometry follows the expected wall intersection model, azimuth indices are ordered correctly, and azimuth spacing matches the configured linspace pattern.

Method:- we perform a single elevation scan pattern at a planar wall perpendicular to boresight and validate beam geometry.

Passing Criteria:- All beams return valid hits on the wall, hit points match analytic intersections, azimuth_index equals 0..N-1, and azimuth samples and spacing match linspace(az_min, az_max, N).

Result:- **Passed**

<img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/0b3adfbe-4e96-4498-aa59-358dbee31db0" />

## Noise test

### 1. Monte Carlo validation of Gaussian range noise statistics against configured parameters.

we collect 10000 samples at fixed ranges, verify empirical standard deviation matches the RSS noise formula within a chi squared confidence interval, and confirm near zero mean error.

Goal: to confirm the stochastic range noise pipeline reproduces the expected distribution: 

zero-mean Gaussian with 

``` math 
\sigma_{total}(r) = sqrt{(\sigma_{fixed})^2 + (k \cdot r)^2}

```
where, $`k`$ is the proportional error coeff, and $`r`$ is the measured range

Passing criteria: At every test range the sample variance ratio $`\frac{s^2}{sigma^2}`$ lies within the 99 percent chi-squared confidence interval and the sample mean error is within $` \frac{3\cdot \sigma}{ \sqrt(N)}`$ of zero.

Result: **Passed**
<img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/8d555dca-d274-4248-8f94-6d603388a9e9" />


### 2. Ensemble Monte Carlo validation of the bias random walk process.

we run N_ensemble=500 independent sensors over a long time series and verify the ensemble variance of bias grows linearly as :

```math

Var(bias(t)) = \sigma_{rw}^2 \cdot t 

```
Goal: Confirm the bias state performs a Wiener process,

```math

Var(bias(T) - bias(0)) = \sigma_{rw}^2 \cdot T 

```

Passing criteria: At every sampled time horizon the ensemble variance ratio Var/expected lies within the 99 percent chi squared confidence interval. Sample trajectories visually exhibit random walk behavior.

Result: **Passed**

<img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/317d5c00-5ea4-4f72-aea5-7f17bac656aa" />

### 3. Deterministic bias drift component

we verify that enabling only bias_drift_rate produces a linearly growing range error whose slope matches the configured drift rate exactly.

Goal: To confirm the deterministic bias drift component,

```math

 \epsilon = \dot b \cdot t 

```

with no stochastic scatter.

Passing criteria: Measured range error at every time step equals $`\dot b \cdot t`$ within machine precision.

**Note :- $` \epsilon `$ is the error, and $`\dot b`$ is the bias drift rate**

Result: *Passed*

<img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/7a1ef9f4-1244-4f85-b696-fae23f8c7a50" />

### 4. Monte Carlo validation of outlier injection rate and magnitude distribution

With outlier probability as 0.10, we collect 10000 samples and verify the empirical outlier rate matches the configured probability. 

Goal: Confirm the outlier injection mechanism fires at the correct rate and draws gross errors from the expected Gaussian distribution.

Passing criteria: Empirical outlier rate lies within a 99 percent binomial confidence interval of  outlier probability. Outlier magnitude mean and standard deviation match outlier bias and outlier std within chi squared and normal confidence bounds.

Result: *Passed*

<img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/8da953a4-e936-4b25-a220-4f095d5137b4" />


Upcoming tests: Beam Geometry 

The test is currently being developed and the progress is slow.



