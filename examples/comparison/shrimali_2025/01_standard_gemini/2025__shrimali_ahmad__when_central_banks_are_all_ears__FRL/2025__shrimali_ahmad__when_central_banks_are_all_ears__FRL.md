Finance Research Letters 86 (2025) 108713

Contents lists available at ScienceDirect
**Finance Research Letters**
journal homepage: www.elsevier.com/locate/frl

# When the central banks are all ears? Examining the communication spillovers over time ✩

Suruchi Shrimali $^{a,1}$, Wasim Ahmad $^{b,*,2}$

$^a$ *Department of Economics & Finance, Birla Institute of Technology & Science, Pilani 333031, Rajasthan, India*
$^b$ *Department of Economic Sciences, Indian Institute of Technology Kanpur, 208016, India*

---

**ARTICLE INFO**

*JEL classification:*
D83
E58
F42

*Keywords:*
Central bank communication
Sentiment spillover
Connectedness analysis
Zero lower bound
Policy uncertainty

**ABSTRACT**

We investigate a new dimension of monetary policy spillover by analysing the sentiment of speeches for twenty central banks between 2000 and 2024. We add to the existing literature by highlighting the temporal relevance of communication spillovers. We observe that the total sentiment spillover varies between 22% to 57%. The findings also reveal key spillover dynamics during global and domestic crises, highlighting the influential role of the Federal Reserve in this network. The macroeconomic conditions in the United States (US) are a major driver of these spillovers. The zero-lower bound in the US accounts for 24% of the variation in overall sentiment spillovers. We contribute to the discussion on the relevance of communication as a policy tool and to the debate on whether central banks are all ears.

---

## 1. Introduction

Information spillover across central banks (CBs) remains a challenging topic of research among researchers despite its domestic appeal. With the emergence of research on CBs using text data, a high-dimensional communication spillover analysis appears to be a possibility. So far, studies have used shadow short rates to exhibit global monetary policy coordination (see Antonakakis et al., 2019; Umar et al., 2024). In this letter, we focus on monetary policy spillovers via CB communication. Unlike Armelius et al. (2020), who explored such spillovers through a static sentiment flow network, we measure the impact and relevance of CB communication under a dynamic set-up for the major central banks. The dynamic set-up helps us measure the strengths and weaknesses of different episodes of global monetary coordination (low interest rates, high inflation, pandemic response) and allows us to answer an important research question: Are central banks all ears? With this, we also link our study with the forward guidance mechanism of central banks and its effectiveness as emphasised by Bowman (2022). Thus, this letter is unique and adds value to the literature by analysing the temporal characteristics of the CB communication spillover.

We quantify communication, the qualitative aspect of monetary policy, using textual analytics by compiling monthly sentiment.$^3$ The sentiments are derived using comprehensive speech data of twenty CBs from February 2000 to March 2024. Further, sentiment spillover indices are constructed by implementing a LASSO-VAR-based connectedness approach. LASSO regularisation facilitates the use of VAR, which is otherwise constrained by the curse of high dimensionality. The dynamic measures are obtained through a rolling window technique. Moreover, we also identify key CBs in the sentiment spillover network by adopting centrality measures. Finally, we examine the impact of US-specific policy factors on sentiment spillovers among the CBs.

The findings reveal interesting time-varying patterns of sentiment spillovers across CBs coinciding with various economic scenarios and events. Moreover, the macroeconomic characteristics of the US are found to explain the dynamics of these spillovers significantly.

## 2. Data and methods

### 2.1. Sample selection and sentiment construction

We collected speech data from the *Bank for International Settlements* (BIS) archive database$^4$ covering February 2000 to March 2024 for 118 CBs. The final sample consists of 20 CBs that had at least one monthly speech on average during this period$^5$ - the United States of America (US), the United Kingdom (GB), Euro Area (EA), Japan (JP), Canada (CA), Australia (AU), France (FR), Italy (IT), Germany (DE), Ireland (IE), Norway (NO), Singapore (SG), Spain (ES), Sweden (SE), Switzerland (CH), Albania (AL), Malaysia (MY), Philippines (PH), India (IN), and South Africa (ZA).

The positive and negative word lists from the finance field Loughran and Mcdonald (2011) dictionary (LM dictionary) are used to extract the average monthly sentiment ($CBSent$). $CBSent$ is a net of positive and negative words over total words, reflecting the CB's net monthly optimism. Its positive value indicates optimism, and its negative value indicates pessimistic sentiment about the CB. A zero value, which implies sentiment neutrality, is assigned when a CB does not deliver any speech in a month. This is because a CB uses speeches as ad-hoc measures, and their occurrence is deliberate$^6$. Further details on sentiment construction can be found in the Online Appendix.

### 2.2. Connectedness indices

$CBSent$ for each CB is used in the LASSO-VAR-based framework$^7$ (see Gabauer et al., 2024) with a lag-length of 1$^8$ to obtain the connectedness matrix ($C^H$),

$$C^H = \begin{bmatrix} \tilde{\phi}_{11}^g & \tilde{\phi}_{12}^g & \dots & \tilde{\phi}_{1m}^g \\ \tilde{\phi}_{21}^g & \tilde{\phi}_{22}^g & \dots & \tilde{\phi}_{2m}^g \\ \vdots & \vdots & \ddots & \vdots \\ \tilde{\phi}_{m1}^g & \tilde{\phi}_{m2}^g & \dots & \tilde{\phi}_{mm}^g \end{bmatrix} \tag{1}$$

Here, $\tilde{\phi}_{ij}^g(H)$ is as pairwise directional communication spillover from CB $j$ to $i$. Meanwhile, the Total Connectedness Index ($TCI$) depicts system-wide spillover, which is computed as $TCI = \frac{1}{m} \sum_{i=1}^m \sum_{j=1, j \neq i}^m \tilde{\phi}_{ij}^g(H)$. The time-varying TCI is obtained by estimating the LASSO-VAR for a rolling window of 70 months. Further, methodological details are provided in the Online Appendix.

### 2.3. Factors affecting overall sentiment spillovers

The Federal Reserve (Fed) is identified as a leader in monetary policy setting (Brusa et al., 2020). To explore whether the US policy environment drives greater sentiment integration across CBs, we estimate the following model,

$$TCI = \beta_0 + \beta_1 ZLB_{US} + \beta_2 UMP_{US} + \beta_3 MPU_{US} + \delta_j ControlVar_j + \epsilon \tag{2}$$

Here, $ZLB_{US}$ and $UMP_{US}$ are constructed using the Shadow Short Rate (SSR) from Krippner (2020). SSR is a market-based measure that captures both conventional and unconventional policy changes. $ZLB_{US}$ is a dummy variable taking the value 1 when SSR is less than 0.125% (binding ZLB rate in the US from Krippner (2020)) and 0 otherwise. Further, the SSR turns negative when a UMP easing exceeds the ZLB. Thus, $UMP_{US} = 0.125 - SSR$, in the case ZLB is binding and zero otherwise, capturing the magnitude of UMP. By construction, it is a non-negative series with a higher magnitude, indicating higher UMP easing. Further, a media-based monetary policy uncertainty ($MPU_{US}$) index from Baker et al. (2016) is used to capture the uncertainty surrounding the Fed's monetary policy. On the other hand, $ControlVar_j$ includes a measure of financial market uncertainty from the CBOE volatility index$^9$ and a measure of global financial markets in the form of the global RMSCI index.

---
✩ This article is part of a Special issue entitled: 'Financial Risk and Uncertainty' published in Finance Research Letters.
\* Corresponding author.
*E-mail addresses:* suruchi.shrimali@pilani.bits-pilani.ac.in, suruchi@iitk.ac.in (S. Shrimali), wasimad@iitk.ac.in (W. Ahmad).
$^1$ Visiting Assistant Professor.
$^2$ Associate Professor.
$^3$ As CBs gradually release private information (Armelius et al., 2020), which is reflected by other CBs with a lag.
$^4$ Accessed from [here](https://www.bis.org/cbspeeches/index.htm) on February 3, 2025.
$^5$ The use of communication as a monetary policy tool is indirect and has been in practice for some recent decades. There are a few countries that have a long history of using communication. However, our selected sample is representative as it includes five major advanced economies, ten advanced economies, and five emerging market economies (based on world bank classification).
$^6$ ARIMA imputation gives similar results.
$^7$ LASSO regularisation overcomes the curse of high-dimensionality in VAR. Thus, this is parallel to the existing literature, which uses correlation-based indices (see Xu and Zhang, 2021) or has high-frequency data to be implemented with VAR and related methods (see Xu, 2018)
$^8$ Based on the Schwarz information criteria.
$^9$ Obtained from FREDdatabase.

---

[Figure 1]
**Fig. 1. TCI with major global and US-specific events.**
The figure shows time-varying total connectedness index (TCI) over the sample period. The red dashed lines mark global and US-specific events.
Source: Authors' Calculations.

## 3. Empirical results

### 3.1. Dynamics of CB communication spillovers

Fig. 1 shows substantial variation in TCI over time, ranging from 22% to 57% with an average of 31%. It started rising with the onset of the global financial crisis (GFC), peaked at 57% in February 2010, and remained above average until the end of 2012. During this period, the zero lower bound (ZLB) was binding, and CBs of major advanced economies heavily relied on communication strategies while adopting unconventional monetary policy (UMP). Further, in May 2013, the US announced a reduction in its quantitative easing program, resulting in global panic known as the ''taper tantrum''. This increased attention to communication and caused a temporary flattening of a falling TCI.

A minor surge in TCI was observed in Fig. 1 after Brexit due to financial market disruptions from ongoing negotiations and thus, increased spillovers. In contrast to the GFC, no significant peak followed COVID-19. This is likely because of its non-financial origin, which created uncertainty around evolving economic conditions. A rise in uncertainty often discourages the use of communication, leading to low sentiment spillovers among CBs.

In addition, the analysis of sentiment spillover reveals significant time variations in the sensitivity (In Strength) and influence (Out Strength) of different CBs. Warmer shades in Fig. 2 indicate periods of higher strength (In and Out). The left panel of Fig. 2 shows that during the GFC, the United States and Japan had notable influence, while the Euro Area's influence increased further during the eurozone crisis. This reflects their roles in recovery and policy measures. In contrast, the United Kingdom showed low influence and sensitivity to other CBs. On the other hand, Canada experienced increased influence and sensitivity during the GFC and euro area crises. This can be due to the co-movement of its long-term interest rates with those of major economies such as the US, France, and Germany, indicating policy alignment and potential sentiment spillover (see Beck et al., 2013).

From Fig. 2, we observe that Ireland's (IE's) engagement increased during the Brexit period due to expected trade disruptions affecting its economy. On the other hand, Norway's (NO's) influence and sensitivity increased after 2014. The low oil prices disrupted NO's oil-dependent economy, leading to speculation about its policy choices and currency interventions. Furthermore, the Swiss National Bank (SNB) adopted unconventional foreign exchange policies and negative interest rates around 2015. A corresponding warmer blocks are observed in Fig. 2 for CH.

Thus, these dynamics reveal that communication spillovers vary with global and domestic events, especially complementing UMP adoption by CBs.

### 3.2. Central nodes in sentiment spillovers

Among the various centrality measures, we use HITS (Hyperlink Induced Topic Search) centrality measures (see Kleinberg, 1999) to identify central nodes in the sentiment spillover network. The benefit of using this measure is that it distinguishes between two roles of the CBs -- sentiment propagator and sentiment receivers by providing the Hub and Authority rankings. A high Hub rank indicates an effective information communicator, while a top Authority represents a CB that is attentive to information from high-ranking Hubs. The authority rank is more similar to the idea of eigenvector or pagerank centrality, whereas a hub rank is similar to the betweenness centrality.

[Figure 2]
**Fig. 2. Time-varying In- and out-strength measures.**
This figure shows the relative time-varying in- and out-strengths (strength of $CB/max(\text{strength of CBs})$) of different central banks. The in strength shows CB's sensitivity to the system, and the out strength shows CB's influence in the system.
Source: Authors' Calculations.

**Table 1**
Top and bottom five HITS centrality ranks.
| Rank | HITS centrality | |
| :--- | :--- | :--- |
| | **Hub** | **Authority** |
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
The table contains HITS centrality measures. A high hub rank represents information communicators, and a high authority rank represents information accumulators.

Table 1 shows EA as the top-ranked authority, highlighting the European Central Bank (ECB) as the most information-sensitive entity in the network. This can be attributed to its role in coordinating multi-country goals. The US, ranked as the number one hub, significantly influences the communication spillover network. It also ranks fifth in authority, indicating strong sensitivity to other CB sentiments. Thus, the US can be classified as the communicator in the network. Additionally, GB, some EU national banks, and some developing economies have low hub and authority rank. This reflects their limited interaction in the sentiment spillover network.

### 3.3. TCI and US policy environment

Fig. 1 shows that TCI spikes during significant global events related to US macroeconomic policies. The US is also a key communicator in the sentiment spillover network, re-emphasising its unique influence and leadership in monetary policy setting. To examine the role of the US policy factors in driving sentiment spillover, we estimate Eq. (2) using OLS with robust standard error.

The regression results in Table 2 show that the US macroeconomic policy significantly affects sentiment spillover between CBs. During the ZLB period in the US, the TCI is notably high, with the $ZLB_{US}$ accounting for 24% of its variation as depicted in column 1 of Table 2. The US effectively used communication as a policy tool to conduct monetary policy when the ZLB constraint was binding. In such a critical situation, there is an increased attention to words spoken among the CBs. This further enhanced overall sentiment spillovers.

**Table 2**
US factors affecting the CB communication spillover.
| Variables | TCI | | | | | |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| | (1) | (2) | (3) | (4) | (5) | (6) |
| $ZLB_{US}$ | 9.842*** | | | 8.561*** | | |
| | (1.258) | | | (1.160) | | |
| $UMP_{US}$ | | 2.965*** | | | 2.751*** | |
| | | (0.478) | | | (0.412) | |
| $MPU_{US}$ | | | -2.698** | | | -6.367*** |
| | | | (1.171) | | | (1.196) |
| $ControlVar$ | No | No | No | Yes | Yes | Yes |
| Constant | 26.944*** | 29.021*** | 44.670*** | 18.172*** | 18.280*** | 49.038*** |
| | (0.413) | (0.656) | (5.935) | (1.472) | (1.451) | (5.759) |
| R-squared | 0.242 | 0.128 | 0.015 | 0.373 | 0.308 | 0.274 |
| F-Stat | 61.25*** | 38.45*** | 5.306** | 34.47*** | 36.12*** | 21.94*** |
| Mean VIF | - | - | - | 1.21 | 1.71 | 1.25 |
| IM-test | 72.88*** | 25.83*** | 9.03** | 102.28*** | 74.03*** | 36.45*** |
The table reports estimate of Eq. (2). This model estimates the effect of the US macroeconomic policy environment on the system-wide sentiment spillovers. In particular, it looks at the effect of ZLB in the US, of the amount of UMP when ZLB is binding, and of the monetary policy uncertainty of the US.
Note: Regression results are obtained with robust standard error. ***, **, and * represent significance at 1%, 5% and 10% level, respectively.
Mean VIF is the mean variance inflation factor for all the independent variables.
IM-test row reports the chi-square statistics for White's heteroskedasticity test with the null hypothesis of homoskedasticity.

$UMP_{US}$ positively affects the TCI as presented in columns 2 and 5 of Table 2. UMP in the US has influence on capital flows, exchange rate stability, and financial markets risk in other countries (Alonso et al., 2024). At ZLB, US policy puts upward pressure on exchange rates and downward pressure on interest rates. In response to financial instability and low policy space, other advanced economies have adopted UMPs, leading to significant capital inflows in emerging markets and a false perception of wealth. As a result, emerging markets increased their reliance on communication to manage financial markets, while advanced economies use it to complement implementation of UMPs.

When analysing policy uncertainties, it is noted from columns 3 and 6 of Table 2 that system-wide communication spillover decreases as $MPU_{US}$ increases. This suggests that noisy macroeconomic data is linked to higher policy uncertainty, making forward guidance a less effective tool (Bowman, 2022). Consequently, there is less reliance on indirect signalling due to increased financial instability risks and reduced attention to these signals. Additionally, low credibility during uncertain times may lead to a deliberate disregard for such communication.

### 3.4. Robustness, sub-sample, and sensitivity analyses

All the following analyses are reported in the Online Appendix.

In the first robustness check, we re-estimate the sentiment by two different approaches -- first, we use a domain fine-tuned RoBERTa model from Pfeifer and Marohl (2023) and second, we integrate valence shifters (such as negators, amplifiers, deamplifiers, and adversative conjunctions) with the LM dictionary, which can change the weight of the sentiment. Then, we re-estimate Eq. (2). The baseline results are robust to an alternate definition of sentiment.

Further, we construct the effective ZLB and UMP measures using the SSR series from Wu and Xia (2016). The variation gives results in line with the baseline results.

Moreover, we add three more global controls to the baseline model - media-based Geopolitical Risk Index (refer Caldara and Iacoviello, 2022), a media-based Oil Policy Uncertainty Index (refer Abiad and Qureshi, 2023), and the World Uncertainty Index (refer Ahir et al., 2022). The baseline results are intact after incorporating these controls.

We also conducted two subsample analyses. First, we re-estimated the spillover indices without the US and estimated the model in Eq. (2). Second, we examined the pre-COVID sample. Our findings show that the US macroeconomic factor remains influential even without its communication in the network. Additionally, stronger results observed when excluding the COVID period.

Further, we examine the sensitivity of TCI to changes in the forecast horizon ($H$) and the rolling window length ($W$). The TCIs remain similar with these alterations.

## 4. Conclusion

When formulating monetary policy, it is essential to consider how communication reflects policy directions in other countries and their potential effects on the domestic economy. The dynamics of sentiment spillovers presented in this letter are crucial for policymakers to understand the monetary policy co-movement through communication channels. We found that communication spillovers vary according to economic conditions. During periods of economic uncertainty, communication spillovers are low, either due to the limited use of sensitive tools, such as communication, or due to the low credibility of words during times of uncertainty. Additionally, the policy environment of the United States plays a critical role in driving the total sentiment spillover from CB communication. However, this study does not examine the spillover variation of individual CBs in different economic states, and the literature can be further extended in this direction. Nonetheless, we conclude that central banks are all ears, especially when policy space is constraint and unconventional policies are implemented.

**CRediT authorship contribution statement**

**Suruchi Shrimali:** Writing - original draft, Visualization, Software, Resources, Methodology, Formal analysis, Data curation, Conceptualization. **Wasim Ahmad:** Writing - review & editing, Supervision, Methodology, Conceptualization.

**Declaration of competing interest**

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

**Appendix A. Supplementary data**

Supplementary material related to this article can be found online at https://doi.org/10.1016/j.frl.2025.108713.

**Data availability**

Data will be made available on request.

**References**

Abiad, A., Qureshi, I.A., 2023. The macroeconomic effects of oil price uncertainty. Energy Econ. 125.
Ahir, H., Bloom, N., Furceri, D., 2022. The World Uncertainty Index. Working Paper 29763, In: Working Paper Series, National Bureau of Economic Research.
Alonso, I., Serrano, P., Vaello-Sebastià, A., 2024. The global spillovers of unconventional monetary policies on tail risks. Financ. Res. Lett. 59.
Antonakakis, N., Gabauer, D., Gupta, R., 2019. International monetary policy spillovers: Evidence from a time-varying parameter vector autoregression. Int. Rev. Financ. Anal. 65.
Armelius, H., Bertsch, C., Hull, I., Zhang, X., 2020. Spread the word: International spillovers from central bank communication. J. Int. Money Financ. 103.
Baker, S.R., Bloom, N., Davis, S.J., 2016. Measuring economic policy uncertainty. Q. J. Econ. 131 (4), 1593-1636.
Beck, M.-K., Hayo, B., Neuenkirch, M., 2013. Central bank communication and correlation between financial markets: Canada and the united states. Int. Econ. Econ. Policy 10 (2), 277-296.
Bowman, M.B., 2022. Forward guidance as a monetary policy tool: considerations for the current economic environment. Board Gov. Reserv. Syst..
Brusa, F., Savor, P., Wilson, M., 2020. One central bank to rule them all. Rev. Financ. 24 (2), 263-304.
Caldara, D., Iacoviello, M., 2022. Measuring geopolitical risk. Am. Econ. Rev. 112 (4), 1194-1225.
Gabauer, D., Gupta, R., Marfatia, H.A., Miller, S.M., 2024. Estimating U.S. housing price network connectedness: Evidence from dynamic elastic net, lasso, and ridge vector autoregressive models. Int. Rev. Econ. Financ. 89, 349-362.
Kleinberg, J.M., 1999. Authoritative sources in a hyperlinked environment. J. ACM 46 (5), 604-632.
Krippner, L., 2020. A note of caution on shadow rate estimates. J. Money Credit. Bank. 52 (4), 951-962.
Loughran, T., Mcdonald, B., 2011. When is a liability not a liability? Textual analysis, dictionaries, and 10-Ks. J. Financ. 66 (1), 35-65.
Pfeifer, M., Marohl, V.P., 2023. CentralBankRoBERTa: A fine-tuned large language model for central bank communications. J. Financ. Data Sci. 9, 100114.
Umar, Z., Bossman, A., Iqbal, N., Teplova, T., 2024. Patterns of unconventional monetary policy spillovers during a systemic crisis. Appl. Econ. 56 (14), 1611-1621.
Wu, J.C., Xia, F.D., 2016. Measuring the macroeconomic impact of monetary policy at the zero lower bound. J. Money Credit. Bank. 48 (2-3), 253-291.
Xu, X., 2018. Intraday price information flows between the CSI300 and futures market: an application of wavelet analysis. Empir. Econ. 54 (3), 1267-1295.
Xu, X., Zhang, Y., 2021. Network analysis of corn cash price comovements. Mach. Learn. Appl. 6, 100140.

## Extracted Images

![Page 1 Image 1](./extracted_images/2025_shrimali_ahmad_when_central_banks_are_all_ears_FRL_page1_img1.png)

![Page 1 Image 2](./extracted_images/2025_shrimali_ahmad_when_central_banks_are_all_ears_FRL_page1_img2.jpeg)

![Page 1 Image 3](./extracted_images/2025_shrimali_ahmad_when_central_banks_are_all_ears_FRL_page1_img3.jpeg)

![Page 3 Image 1](./extracted_images/2025_shrimali_ahmad_when_central_banks_are_all_ears_FRL_page3_img1.jpeg)

![Page 4 Image 1](./extracted_images/2025_shrimali_ahmad_when_central_banks_are_all_ears_FRL_page4_img1.jpeg)