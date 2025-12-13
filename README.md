## Credit Scoring Business Understanding

### 1. Basel II’s Influence on Model Interpretability

Basel II places strong emphasis on advanced, risk-sensitive credit risk measurement, particularly through the Internal Ratings-Based (IRB) approach. This regulatory shift requires credit risk models to be interpretable, transparent, and well-documented, not merely predictive.

**Key Basel II Requirements and Their Impact**

- **Supervisory Review (Pillar 2)**  
  Regulators must be able to validate internal models used to calculate Probability of Default (PD), Loss Given Default (LGD), and Exposure at Default (EAD). This validation is not feasible with black-box models, making interpretability and transparency essential.

- **Market Discipline (Pillar 3)**  
  Banks are required to publicly disclose information about their risk profiles and capital adequacy. To maintain credibility and allow investors and analysts to compare risk measures, the underlying models must be understandable and produce economically plausible results.

- **Internal Risk Management**  
  Effective risk management requires bank management, internal auditors, and risk officers to understand the drivers of model outputs. Interpretable and well-documented models ensure that:
  - Human judgment can be applied appropriately  
  - Assumptions and limitations are clearly defined  
  - Model validation and updates can be performed systematically  

**Summary**  
Basel II shifted regulatory focus from standardized calculations to internal, risk-sensitive modeling. This flexibility is granted only with a corresponding burden of proof: banks must demonstrate through documentation and interpretability that their models are robust, reliable, and suitable for determining regulatory capital.

---

### 2. Necessity and Risks of Using a Proxy Variable for Default Prediction

In credit risk modeling, the true event of default—often defined legally (e.g., bankruptcy or 90+ days past due)—is frequently unobserved, delayed, or too rare in available datasets. As a result, proxy variables are commonly used to approximate default behavior.

#### 2.1 Why a Proxy Variable Is Necessary

- **Data Scarcity and Rarity**  
  True default events occur infrequently, leading to highly imbalanced datasets and insufficient data for effective model training.

- **Observation Lag and Censoring**  
  The actual default event may occur long after the observation window ends or may never be observed if the loan is sold, prepaid, or otherwise exits the portfolio.

- **Definition Ambiguity**  
  For certain asset classes, such as private equity or retail lending, a consistent legal definition of default may not be available across all borrowers.

---

#### 2.2 Potential Business Risks of Using a Proxy Variable

Using an imperfect proxy instead of the true default event introduces systematic risks into business and regulatory decision-making.

- **Model Inaccuracy (Proxy Bias)**  
  The proxy variable is not perfectly correlated with true default. A model may predict the proxy event accurately while systematically overestimating or underestimating actual default risk.

- **Capital Miscalculation Risk**  
  Under Basel II, inaccurate Probability of Default estimates lead to incorrect regulatory capital requirements. This may result in holding too little capital (regulatory and solvency risk) or too much capital (reduced profitability and competitiveness).

- **Adverse Business Decisions**  
  Credit decisions based on flawed proxies may lead to rejecting creditworthy borrowers (false positives) or approving high-risk borrowers (false negatives).

- **Regulatory and Fair Lending Risk**  
  If a proxy is correlated with protected characteristics such as age, location, or socioeconomic status, it may be viewed as an indirect proxy for discrimination, exposing the institution to legal and reputational risk.

- **Lack of Interpretability**  
  If the proxy event is abstract or poorly linked to actual default, it becomes difficult to explain credit decisions to customers, regulators, and internal auditors.

---

### 3. Key Trade-Off: Simple vs. Complex Models in Regulated Finance

In regulated financial environments, the choice of credit risk model involves a trade-off between predictive accuracy and trust, interpretability, and regulatory compliance.

#### 3.1 Simple, Interpretable Models (Logistic Regression with Weight of Evidence)

Logistic regression with Weight of Evidence (WoE) transformations is the traditional standard for credit scorecards. It relies on linear relationships after monotonic transformation of input features.

**Advantages**
- Highly transparent and easy to audit  
- Strong regulatory acceptance under Basel II Pillar 2  
- Stable over time and less prone to overfitting  
- Produces clear, legally compliant adverse-action explanations  

**Limitations**
- Lower predictive performance compared to complex, non-linear models  
- Requires extensive manual feature engineering and binning  
- Assumes monotonic or linear relationships after transformation  

**Model Form**  
LogOdds = β0 + Σ(βi × WoEi)

---

### 4. Overall Business Perspective

Under Basel II, predictive accuracy alone is insufficient. Credit risk models must be interpretable, auditable, and defensible. Consequently, financial institutions often prioritize transparent, well-documented models or apply strict explainability controls when deploying more complex modeling techniques.
