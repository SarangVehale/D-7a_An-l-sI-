-- inferential_analysis.sql

-- 1. Age: mean, standard deviation, sample size
.output age_summary.csv
SELECT
  COUNT("Your Age (In Completed Years)")                             AS n,
  ROUND(AVG("Your Age (In Completed Years)"), 2)                      AS mean_age,
  ROUND(
    SQRT(
      (
        SUM("Your Age (In Completed Years)" * "Your Age (In Completed Years)")
        - SUM("Your Age (In Completed Years)") * SUM("Your Age (In Completed Years)") / COUNT(*)
      )
      / (COUNT(*) - 1)
    ),
    2
  )                                                                  AS sd_age
FROM survery_responses;

-- 2. Demographic frequency tables
.output gender_distribution.csv
SELECT
  "What Gender do you identify as?"                                 AS gender,
  COUNT(*)                                                          AS count
FROM survery_responses
GROUP BY gender;

.output education_distribution.csv
SELECT
  "Your Highest Education Level: "                                 AS education,
  COUNT(*)                                                         AS count
FROM survery_responses
GROUP BY education;

.output employment_distribution.csv
SELECT
  "Your Current Employment Status: "                               AS employment_status,
  COUNT(*)                                                         AS count
FROM survery_responses
GROUP BY employment_status;

-- 3. Key question frequencies
.output heard_abuse_freq.csv
SELECT
  "Have you ever heard of cases where men were sexually abused?  "  AS heard_abuse,
  COUNT(*)                                                         AS count
FROM survery_responses
GROUP BY heard_abuse;

.output support_freq.csv
SELECT
  "Do you think support from the social circle can help reduce stigma and encourage male survivors to speak up?  "
                                                                    AS support_social_circle,
  COUNT(*)                                                         AS count
FROM survery_responses
GROUP BY support_social_circle;

-- 4. Welch’s t‐test components for age difference by gender
.output ttest_age_by_gender.csv
WITH stats AS (
  SELECT
    "What Gender do you identify as?"       AS gender,
    COUNT(*)                                AS n,
    AVG("Your Age (In Completed Years)")    AS mean_age,
    -- sample variance
    (
      SUM("Your Age (In Completed Years)" * "Your Age (In Completed Years)")
      - SUM("Your Age (In Completed Years)") * SUM("Your Age (In Completed Years)") / COUNT(*)
    )
    / (COUNT(*) - 1)                         AS var_age
  FROM survery_responses
  GROUP BY gender
)
SELECT
  g1.gender                                               AS group1,
  g1.n                                                    AS n1,
  ROUND(g1.mean_age,2)                                    AS mean1,
  ROUND(g1.var_age,2)                                     AS var1,
  g2.gender                                               AS group2,
  g2.n                                                    AS n2,
  ROUND(g2.mean_age,2)                                    AS mean2,
  ROUND(g2.var_age,2)                                     AS var2,
  -- Welch’s t statistic
  ROUND(
    (g1.mean_age - g2.mean_age)
    / SQRT(g1.var_age/g1.n + g2.var_age/g2.n),
  4)                                                        AS t_statistic,
  -- Welch–Satterthwaite degrees of freedom
  ROUND(
    (
      (g1.var_age/g1.n + g2.var_age/g2.n) * (g1.var_age/g1.n + g2.var_age/g2.n)
    )
    / (
      (g1.var_age*g1.var_age) / ( (g1.n * g1.n)*(g1.n - 1) )
      + (g2.var_age*g2.var_age) / ( (g2.n * g2.n)*(g2.n - 1) )
    ),
  2)                                                        AS df_approx
FROM stats g1
JOIN stats g2
  ON g1.gender <> g2.gender;

-- 5. Chi‐square: Gender × Heard‐Abuse
.output chi2_gender_heard.csv
WITH
  obs AS (
    SELECT
      "What Gender do you identify as?"                         AS gender,
      "Have you ever heard of cases where men were sexually abused?  " AS heard,
      COUNT(*)                                                  AS observed
    FROM survery_responses
    GROUP BY gender, heard
  ),
  marg_gender AS (
    SELECT gender, SUM(observed) AS total_gender
    FROM obs GROUP BY gender
  ),
  marg_heard AS (
    SELECT heard, SUM(observed) AS total_heard
    FROM obs GROUP BY heard
  ),
  tot AS (SELECT SUM(observed) AS grand_total FROM obs),
  expected AS (
    SELECT
      o.gender,
      o.heard,
      (mg.total_gender * mh.total_heard) * 1.0 / t.grand_total AS expected
    FROM obs o
    JOIN marg_gender mg ON o.gender = mg.gender
    JOIN marg_heard mh  ON o.heard  = mh.heard
    CROSS JOIN tot t
  ),
  comps AS (
    SELECT
      o.gender,
      o.heard,
      o.observed,
      e.expected,
      ((o.observed - e.expected)*(o.observed - e.expected) / e.expected) AS chi_comp
    FROM obs o
    JOIN expected e
      ON o.gender = e.gender
     AND o.heard  = e.heard
  )
SELECT
  ROUND(SUM(chi_comp), 4)                                     AS chi_square,
  -- df = (r-1)*(c-1) = (2-1)*(2-1) = 1 for 2×2
  1                                                           AS df
FROM comps;

-- 6. Chi‐square: Gender × Support‐Social‐Circle
.output chi2_gender_support.csv
WITH
  obs2 AS (
    SELECT
      "What Gender do you identify as?"                                              AS gender,
      "Do you think support from the social circle can help reduce stigma and encourage male survivors to speak up?  "
                                                                                      AS support,
      COUNT(*)                                                                       AS observed
    FROM survery_responses
    GROUP BY gender, support
  ),
  mg2 AS (
    SELECT gender, SUM(observed) AS total_gender FROM obs2 GROUP BY gender
  ),
  ms2 AS (
    SELECT support, SUM(observed) AS total_support FROM obs2 GROUP BY support
  ),
  tot2 AS (SELECT SUM(observed) AS grand_total FROM obs2),
  exp2 AS (
    SELECT
      o2.gender,
      o2.support,
      (mg2.total_gender * ms2.total_support) * 1.0 / tot2.grand_total AS expected
    FROM obs2 o2
    JOIN mg2  ON o2.gender  = mg2.gender
    JOIN ms2  ON o2.support = ms2.support
    CROSS JOIN tot2
  ),
  comp2 AS (
    SELECT
      o2.gender,
      o2.support,
      o2.observed,
      e2.expected,
      ((o2.observed - e2.expected)*(o2.observed - e2.expected) / e2.expected) AS chi_comp
    FROM obs2 o2
    JOIN exp2 e2
      ON o2.gender  = e2.gender
     AND o2.support = e2.support
  )
SELECT
  ROUND(SUM(chi_comp), 4)                                     AS chi_square,
  -- df = (r-1)*(c-1) = (2-1)*(3-1) = 2
  2                                                           AS df
FROM comp2;

-- Reset output to STDOUT
.output stdout

