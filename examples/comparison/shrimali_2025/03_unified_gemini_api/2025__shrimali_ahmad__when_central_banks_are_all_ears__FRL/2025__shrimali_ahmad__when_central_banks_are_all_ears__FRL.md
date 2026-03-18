This is a transcription of the research paper shown in the image:

---

**Journal:** Finance Research Letters 86 (2025) 108713
**Journal Homepage:** [www.elsevier.com/locate/frl](http://www.elsevier.com/locate/frl)

# When the central banks are all ears? Examining the communication spillovers over time

**Authors:**
*   **Suruchi Shrimali** $^a, 1$
*   **Wasim Ahmad** $^b, *, 2$

$^a$ *Department of Economics & Finance, Birla Institute of Technology & Science, Pilani 333031, Rajasthan, India*
$^b$ *Department of Economic Sciences, Indian Institute of Technology Kanpur, 208016, India*

---

### ARTICLE INFO
**JEL classification:**
D83, E58, F42

**Keywords:**
*   Central bank communication
*   Sentiment spillover
*   Connectedness analysis
*   Zero lower bound
*   Policy uncertainty

---

### ABSTRACT
We investigate a new dimension of monetary policy spillover by analysing the sentiment of speeches for twenty central banks between 2000 and 2024. We add to the existing literature by highlighting the temporal relevance of communication spillovers. We observe that the total sentiment spillover varies between 22% to 57%. The findings also reveal key spillover dynamics during global and domestic crises, highlighting the influential role of the Federal Reserve in this network. The macroeconomic conditions in the United States (US) are a major driver of these spillovers. The zero-lower bound in the US accounts for 24% of the variation in overall sentiment spillovers. We contribute to the discussion on the relevance of communication as a policy tool and to the debate on whether central banks are all ears.

---

### 1. Introduction
Information spillover across central banks (CBs) remains a challenging topic of research among researchers despite its domestic appeal. With the emergence of research on CBs using text data, a high-dimensional communication spillover analysis appears to be a possibility. So far, studies have used shadow short rates to exhibit global monetary policy coordination (see Antonakakis et al., 2019; Umar et al., 2024). In this letter, we focus on monetary policy spillovers via CB communication. Unlike Armelius et al. (2020), who explored such spillovers through a static sentiment flow network, we measure the impact and relevance of CB communication under a dynamic set-up for the major central banks. The dynamic set-up helps us measure the strengths and weaknesses of different episodes of global monetary coordination (low interest rates, high inflation, pandemic response) and allows us to answer an important research question: Are central banks all ears? With this, we also link our study with the forward guidance mechanism of central banks and its effectiveness as emphasised by Bowman (2022). Thus, this letter is unique and adds value to the literature by analysing the temporal characteristics of the CB communication spillover.

We quantify communication, the qualitative aspect of monetary policy, using textual analytics by compiling monthly sentiment.$^3$ The sentiments are derived using comprehensive speech data of twenty CBs from February 2000 to March 2024. Further, sentiment spillover indices are constructed by implementing a LASSO-VAR-based connectedness approach. LASSO regularisation facilitates the use of VAR, which is otherwise constrained by the curse of high dimensionality. The dynamic measures are obtained through a...

---

### Footnotes & Publication Details
*   **Special Issue:** This article is part of a Special issue entitled: ‘Financial Risk and Uncertainty’ published in Finance Research Letters.
*   **Corresponding author:** Wasim Ahmad.
*   **E-mail addresses:** suruchi.shrimali@pilani.bits-pilani.ac.in, suruchi@iitk.ac.in (S. Shrimali), wasimad@iitk.ac.in (W. Ahmad).
*   $^1$ Visiting Assistant Professor.
*   $^2$ Associate Professor.
*   $^3$ As CBs gradually release private information (Armelius et al., 2020), which is reflected by other CBs with a lag.
*   **DOI:** [https://doi.org/10.1016/j.frl.2025.108713](https://doi.org/10.1016/j.frl.2025.108713)
*   **Received:** 3 September 2025; **Revised:** 4 October 2025; **Accepted:** 14 October 2025
*   **Available online:** 17 October 2025
*   **Copyright:** 1544-6123/© 2025 Elsevier Inc. All rights reserved, including those for text and data mining, AI training, and similar technologies.

---

*S. Shrimali and W. Ahmad*
*Finance Research Letters 86 (2025) 108713*

rolling window technique. Moreover, we also identify key CBs in the sentiment spillover network by adopting centrality measures. Finally, we examine the impact of US-specific policy factors on sentiment spillovers among the CBs.

The findings reveal interesting time-varying patterns of sentiment spillovers across CBs coinciding with various economic scenarios and events. Moreover, the macroeconomic characteristics of the US are found to explain the dynamics of these spillovers significantly.

### 2. Data and methods

#### 2.1. Sample selection and sentiment construction

We collected speech data from the *Bank for International Settlements* (BIS) archive database$^4$ covering February 2000 to March 2024 for 118 CBs. The final sample consists of 20 CBs that had at least one monthly speech on average during this period$^5$ - the United States of America (US), the United Kingdom (GB), Euro Area (EA), Japan (JP), Canada (CA), Australia (AU), France (FR), Italy (IT), Germany (DE), Ireland (IE), Norway (NO), Singapore (SG), Spain (ES), Sweden (SE), Switzerland (CH), Albania (AL), Malaysia (MY), Philippines (PH), India (IN), and South Africa (ZA).

The positive and negative word lists from the finance field Loughran and Mcdonald (2011) dictionary (LM dictionary) are used to extract the average monthly sentiment ($CBSent$). $CBSent$ is a net of positive and negative words over total words, reflecting the CB's net monthly optimism. Its positive value indicates optimism, and its negative value indicates pessimistic sentiment about the CB. A zero value, which implies sentiment neutrality, is assigned when a CB does not deliver any speech in a month. This is because a CB uses speeches as ad-hoc measures, and their occurrence is deliberate$^6$. Further details on sentiment construction can be found in the Online Appendix.

#### 2.2. Connectedness indices

$CBSent$ for each CB is used in the LASSO-VAR-based framework$^7$ (see Gabauer et al., 2024) with a lag-length of 1$^8$ to obtain the connectedness matrix ($C^H$),

$$C^H = \begin{bmatrix} \tilde{\tau}_{11}^g & \tilde{\tau}_{12}^g & \dots & \tilde{\tau}_{1m}^g \\ \tilde{\tau}_{21}^g & \tilde{\tau}_{22}^g & \dots & \tilde{\tau}_{2m}^g \\ \vdots & \vdots & \ddots & \vdots \\ \tilde{\tau}_{m1}^g & \tilde{\tau}_{m2}^g & \dots & \tilde{\tau}_{mm}^g \end{bmatrix} \eqno(1)$$

Here, $\tilde{\tau}_{ij}^g(H)$ is as pairwise directional communication spillover from CB $j$ to $i$. Meanwhile, the Total Connectedness Index ($TCI$) depicts system-wide spillover, which is computed as $TCI = \frac{1}{m} \sum_{i=1}^m \sum_{j=1, j \neq i}^m \tilde{\tau}_{ij}^g(H)$. The time-varying TCI is obtained by estimating the LASSO-VAR for a rolling window of 70 months. Further, methodological details are provided in the Online Appendix.

#### 2.3. Factors affecting overall sentiment spillovers

The Federal Reserve (Fed) is identified as a leader in monetary policy setting (Brusa et al., 2020). To explore whether the US policy environment drives greater sentiment integration across CBs, we estimate the following model,

$$TCI = \beta_0 + \beta_1 ZLB_{US} + \beta_2 UMP_{US} + \beta_3 MPU_{US} + \delta_j ControlVar_j + \epsilon \eqno(2)$$

Here, $ZLB_{US}$ and $UMP_{US}$ are constructed using the Shadow Short Rate (SSR) from Krippner (2020). SSR is a market-based measure that captures both conventional and unconventional policy changes. $ZLB_{US}$ is a dummy variable taking the value 1 when SSR is less than 0.125% (binding ZLB rate in the US from Krippner (2020)) and 0 otherwise. Further, the SSR turns negative when a UMP easing exceeds the ZLB. Thus, $UMP_{US} = 0.125 - SSR$, in the case ZLB is binding and zero otherwise, capturing the magnitude of UMP. By construction, it is a non-negative series with a higher magnitude, indicating higher UMP easing. Further, a media-based monetary policy uncertainty ($MPU_{US}$) index from Baker et al. (2016) is used to capture the uncertainty surrounding the Fed's monetary policy. On the other hand, $ControlVar_j$ includes a measure of financial market uncertainty from the CBOE volatility index$^9$ and a measure of global financial markets in the form of the global RMSCI index.

***

$^4$ Accessed from [here](https://www.bis.org/speeches/index.htm) on February 3, 2025.
$^5$ The use of communication as a monetary policy tool is indirect and has been in practice for some recent decades. There are a few countries that have a long history of using communication. However, our selected sample is representative as it includes five major advanced economies, ten advanced economies, and five emerging market economies (based on world bank classification).
$^6$ ARIMA imputation gives similar results.
$^7$ LASSO regularisation overcomes the curse of high-dimensionality in VAR. Thus, this is parallel to the existing literature, which uses correlation-based indices (see Xu and Zhang, 2021) or has high-frequency data to be implemented with VAR and related methods (see Xu, 2018)
$^8$ Based on the Schwarz information criteria.
$^9$ Obtained from FREDdatabase.

<p align="center">2</p>

---

This document page is from a research paper titled **"Finance Research Letters 86 (2025) 108713"** by **S. Shrimali and W. Ahmad**. It details empirical results regarding the dynamics of central bank (CB) communication spillovers using a Total Connectedness Index (TCI).

### **Figure 1: TCI with major global and US-specific events**
The graph illustrates the time-varying Total Connectedness Index (TCI) from approximately April 2005 to October 2024.
*   **Y-axis:** TCI values ranging from 20 to over 50.
*   **X-axis:** Time in months (MM-YY format).
*   **Key Findings from the Graph:**
    *   The TCI fluctuates between **22% and 57%**, with an **average of 31%**.
    *   A significant surge is visible starting with the **Global Financial Crisis (GFC)**, reaching a peak of **57% in February 2010**.
    *   High levels of connectedness remained until the end of 2012, a period characterized by the **Lehman Brothers collapse**, the **Eurozone crisis**, and the **US credit rating downgrade**.
    *   The **"Taper Tantrum"** in May 2013 led to a temporary flattening and subsequent decline in TCI.
    *   Other marked events include **Brexit** (showing a minor surge), **Covid-19** (showing no significant peak), and the beginning of **FOMC Tightening** in 2022.

### **Section 3.1: Dynamics of CB communication spillovers**
*   **GFC Period:** The high TCI during and after the GFC is attributed to central banks relying heavily on communication strategies while adopting unconventional monetary policy (UMP) as interest rates hit the zero lower bound (ZLB).
*   **Taper Tantrum:** The announcement of reduced quantitative easing in the US caused global panic, increasing attention to communication but temporarily flattening the TCI.
*   **Brexit vs. COVID-19:** Brexit caused a minor surge due to financial market disruptions. In contrast, COVID-19 did not see a significant peak in TCI, likely because of its non-financial origin, which led to lower sentiment spillovers among CBs.
*   **Regional Observations:**
    *   The **US and Japan** had notable influence during the GFC.
    *   The **Euro Area's** influence increased during the eurozone crisis.
    *   **Canada** showed increased sensitivity and influence during both the GFC and eurozone crises due to policy alignment with major economies.
    *   **Ireland** saw increased engagement during Brexit.
    *   **Norway's** influence rose after 2014 due to oil price disruptions.
    *   The **Swiss National Bank (SNB)** showed increased activity around 2015 due to unconventional foreign exchange policies.

### **Section 3.2: Central nodes in sentiment spillovers**
The authors use **HITS (Hyperlink Induced Topic Search)** centrality measures to identify key players in the sentiment spillover network:
*   **Hub:** Represents a "sentiment propagator" or an effective information communicator.
*   **Authority:** Represents a "sentiment receiver" or a central bank that is highly attentive to information from high-ranking Hubs.
*   This distinction helps identify which central banks are leading the narrative and which are primarily reacting to it.

---

Based on the provided image, here is the extracted information:

### **Figure 2: Time-varying In- and out-strength measures**
The image displays two heatmaps representing the relative time-varying in-strength and out-strength of various central banks (CBs) from March 2006 to March 2024.

*   **In-Strength:** Represents a central bank's sensitivity to the system.
*   **Out-Strength:** Represents a central bank's influence on the system.
*   **Nodes (Central Banks):** ZA, IN, PH, MY, AL, CH, SE, ES, SG, NO, IE, AU, DE, IT, FR, CA, GB, JP, EA, US.
*   **Color Scale:** Ranges from blue (low strength, ~0.25) to red (high strength, 1.00).

---

### **Table 1: Top and bottom five HITS centrality ranks**
The table ranks central banks based on their HITS (Hyperlink-Induced Topic Search) centrality measures. A high **hub rank** indicates an information communicator, while a high **authority rank** indicates an information accumulator.

| Rank | Hub (Communicator) | Authority (Accumulator) |
| :--- | :--- | :--- |
| 1 | US | EA |
| 2 | CA | CH |
| 3 | IE | JP |
| 4 | JP | SE |
| 5 | SE | US |
| 16 | FR | FR |
| 17 | AL | AL |
| 18 | ZA | ZA |
| 19 | SG | GB |
| 20 | GB | SG |

---

### **Key Findings from the Text**
*   **European Central Bank (EA):** Ranked as the top authority, highlighting it as the most information-sensitive entity in the network due to its role in coordinating multi-country goals.
*   **United States (US):** Ranked as the number one hub, significantly influencing the communication spillover network. It also ranks fifth in authority, indicating strong sensitivity to other CB sentiments.
*   **Low-Ranked Entities:** Great Britain (GB), some EU national banks, and some developing economies have low hub and authority ranks, reflecting limited interaction in the sentiment spillover network.
*   **US Policy Impact:** The text notes that US macroeconomic policy significantly affects sentiment spillover. During the Zero Lower Bound (ZLB) period in the US, the Total Connectedness Index (TCI) was notably high, with the $ZLB_{US}$ accounting for 24% of its variation.

---

Based on the provided image from a research paper titled "Finance Research Letters 86 (2025) 108713," here is a summary of the key information and data presented.

### **Table 2: US factors affecting the CB communication spillover**

This table reports the estimated effects of the US macroeconomic policy environment on system-wide sentiment spillovers, measured by the Total Connectedness Index (TCI).

| Variables | (1) | (2) | (3) | (4) | (5) | (6) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$ZLB_{US}$** | 9.842*** (1.258) | | | 8.561*** (1.160) | | |
| **$UMP_{US}$** | | 2.965*** (0.478) | | | 2.751*** (0.412) | |
| **$MPU_{US}$** | | | -2.698** (1.171) | | | -6.367*** (1.196) |
| **Control Var** | No | No | No | Yes | Yes | Yes |
| **Constant** | 26.944*** (0.413) | 29.021*** (0.656) | 44.670*** (5.935) | 18.172*** (1.472) | 18.280*** (1.451) | 49.038*** (5.759) |
| **R-squared** | 0.242 | 0.128 | 0.015 | 0.373 | 0.308 | 0.274 |
| **F-Stat** | 61.25*** | 38.45*** | 5.306** | 34.47*** | 36.12*** | 21.94*** |
| **Mean VIF** | - | - | - | 1.21 | 1.71 | 1.25 |
| **IM-test** | 72.88*** | 25.83*** | 9.03** | 102.28*** | 74.03*** | 36.45*** |

*Note: Robust standard errors are in parentheses. Significance levels: \*\*\* p<0.01, \*\* p<0.05, \* p<0.1.*

---

### **Key Findings from the Analysis**

The text accompanying the table provides further context for these results:

*   **Zero Lower Bound ($ZLB_{US}$):** The positive and significant coefficients in models (1) and (4) indicate that when the US interest rates are at the zero lower bound, there is increased attention to central bank communication, which enhances overall sentiment spillovers.
*   **Unconventional Monetary Policy ($UMP_{US}$):** As shown in models (2) and (5), US unconventional monetary policy has a positive and significant impact on TCI. The authors suggest that US UMP influences global capital flows and financial markets, leading other countries to rely more heavily on communication to manage these external pressures.
*   **Monetary Policy Uncertainty ($MPU_{US}$):** Models (3) and (6) show a negative and significant relationship. This suggests that higher uncertainty regarding US monetary policy reduces system-wide communication spillovers. High uncertainty makes forward guidance less effective, leading to a decreased reliance on indirect signaling.

### **Robustness and Sensitivity Checks**

The authors conducted several additional analyses to confirm the stability of their findings:
*   **Sentiment Measurement:** They re-estimated sentiment using alternative methods, including a domain-specific RoBERTa model and a dictionary-based approach with valence shifters.
*   **Alternative Variables:** They used different measures for ZLB and UMP based on the Shadow Short Rate (SSR) series.
*   **Additional Controls:** The results remained intact after adding global controls like Geopolitical Risk, Oil Policy Uncertainty, and World Uncertainty indices.
*   **Sub-sample Analysis:** The findings were consistent when excluding the US from the network and when focusing on the pre-COVID-19 period.
*   **Sensitivity Analysis:** The TCI results were found to be similar across different forecast horizons and rolling window lengths.

---

Based on the image provided, here is the transcribed list of references:

**References**

*   Abiad, A., Qureshi, I.A., 2023. The macroeconomic effects of oil price uncertainty. Energy Econ. 125.
*   Ahir, H., Bloom, N., Furceri, D., 2022. The World Uncertainty Index. Working Paper 29763, In: Working Paper Series, National Bureau of Economic Research.
*   Alonso, I., Serrano, P., Vaello-Sebastià, A., 2024. The global spillovers of unconventional monetary policies on tail risks. Financ. Res. Lett. 59.
*   Antonakakis, N., Gabauer, D., Gupta, R., 2019. International monetary policy spillovers: Evidence from a time-varying parameter vector autoregression. Int. Rev. Financ. Anal. 65.
*   Armelius, H., Bertsch, C., Hull, I., Zhang, X., 2020. Spread the word: International spillovers from central bank communication. J. Int. Money Financ. 103.
*   Baker, S.R., Bloom, N., Davis, S.J., 2016. Measuring economic policy uncertainty. Q. J. Econ. 131 (4), 1593–1636.
*   Beck, M.-K., Hayo, B., Neuenkirch, M., 2013. Central bank communication and correlation between financial markets: Canada and the united states. Int. Econ. Econ. Policy 10 (2), 277–296.
*   Bowman, M.B., 2022. Forward guidance as a monetary policy tool: considerations for the current economic environment. Board Gov. Reserv. Syst..
*   Brusa, F., Savor, P., Wilson, M., 2020. One central bank to rule them all. Rev. Financ. 24 (2), 263–304.
*   Caldara, D., Iacoviello, M., 2022. Measuring geopolitical risk. Am. Econ. Rev. 112 (4), 1194–1225.
*   Gabauer, D., Gupta, R., Marfatia, H.A., Miller, S.M., 2024. Estimating U.S. housing price network connectedness: Evidence from dynamic elastic net, lasso, and ridge vector autoregressive models. Int. Rev. Econ. Financ. 89, 349–362.
*   Kleinberg, J.M., 1999. Authoritative sources in a hyperlinked environment. J. ACM 46 (5), 604–632.
*   Krippner, L., 2020. A note of caution on shadow rate estimates. J. Money Credit. Bank. 52 (4), 951–962.
*   Loughran, T., Mcdonald, B., 2011. When is a liability not a liability? Textual analysis, dictionaries, and 10-Ks. J. Financ. 66 (1), 35–65.
*   Pfeifer, M., Marohl, V.P., 2023. CentralBankRoBERTa: A fine-tuned large language model for central bank communications. J. Financ. Data Sci. 9, 100114.
*   Umar, Z., Bossman, A., Iqbal, N., Teplova, T., 2024. Patterns of unconventional monetary policy spillovers during a systemic crisis. Appl. Econ. 56 (14), 1611–1621.
*   Wu, J.C., Xia, F.D., 2016. Measuring the macroeconomic impact of monetary policy at the zero lower bound. J. Money Credit. Bank. 48 (2–3), 253–291.
*   Xu, X., 2018. Intraday price information flows between the CSI300 and futures market: an application of wavelet analysis. Empir. Econ. 54 (3), 1267–1295.
*   Xu, X., Zhang, Y., 2021. Network analysis of corn cash price comovements. Mach. Learn. Appl. 6, 100140.