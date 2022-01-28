clear all

cd "C:\Users\Helmuth\OneDrive\Studies\Graduate\University of Milan\M.Sc. Data Science & Economics\1st Year\Microeconometrics & Causal Inference\Empirical Project - US Elections\Project Folder (Dropbox)"
use project.dta

*Set panel data
encode state, gen(n_state)
xtset n_state year, yearly

* label variables
lab var state "U.S State"
lab var year "Election year"
lab var Votes "Democratic share of presidential vote"
lab var i "1 [-1] if Dem [Rep] presidential incumbent"
lab var DPER "1 [-1] {0} Dem [Rep] presidential incumbent {not} running again"
lab var DUR "Duration party in charge"
lab var G "per capita real GDP growth rate Q1-Q3 og the election year"
lab var P "Inflation rate (first 15Q of admin)"
lab var Z "No. quarters GDP growt>3.2% during (first 15Q of admin)"
lab var tradition "Party-winning trend of the last 4 election periods"
lab var popdensity "No. of persons for squared km"
lab var unemployment "Unemployment rate growth Q1-Q3 of the election year"
lab var PerIncome "Personal Income growth rate Q1-Q3 of the election year"
lab var Militexp "Military expenditure growth rate Q1-Q3 of the election year"
lab var TurnOut "No. of voters as a percentage of total possible voters"
lab var EV2012 "Electoral Votes per state for 2012"
lab var EV2020 "Electoral Votes per state for 2020"

* label values of categorical variables
lab def incumbent 1 "Dem incumbent" -1 "Rep incumbent" 0 "Third party"
lab val i incumbent

label define DPER 1 "Dem running again" -1 "Rep running again" 0 "Incumbent not running again"
lab val DPER DPER

lab def DUR 0 "Dem or Rep party 1 term" /*
*/ 1 "Dem party 2 terms"  2 "Dem party 6 terms"/*
*/ -1 "Rep party 2 terms"    -2 "Rep party 6 terms"
lab val DUR DUR

* Presidents 
gen president = "WILSON" 
lab var president "President name"
replace president = "REAGAN" if year==1980
replace president = "REAGAN" if year==1984
replace president = "BUSH" if year==1988
replace president = "CLINTON" if year==1992
replace president = "CLINTON" if year==1996
replace president = "BUSH (W.)" if year==2000
replace president = "BUSH (W.)" if year==2004
replace president = "OBAMA" if year==2008
replace president = "OBAMA" if year==2012
replace president = "TRUMP" if year==2016

* Interactions
gen G_i = G*i
gen P_i = P*i
gen Z_i = Z*i
label var G_i "g x i"
label var P_i "p x i"
label var Z_i "z x i"

*Fixed effects regression
reg Votes G_i P_i Z_i DPER DUR i i.n_state if year<2020, robust
est sto reg_2020

*Tests of significance
testparm i.n_state
test G_i P_i Z_i DPER DUR i

*Regression to compare results with lasso DS
tab state, gen(D)
reg Votes G_i P_i Z_i DPER DUR i D2-D51 if year<2020, robust
est sto reg_2020

*Endogeneity Test 2020
*xtreg Votes G_i P_i Z_i DPER DUR i if year<2020, fe robust


****------------- 2020 PREDICTION------------- ****
*Prediction variables
predict vphat if year==2020, xb
predict uhat if year==2020, resid

*Prediction (vphat) and associated residuals (uhat) of the 2020 election per state
list year state Votes vphat uhat if year==2020

graph bar uhat if year==2020, over (state)
*FIX X-AXIS FOR THE REPORT******************
 * If we want to see a State voting trendline
 * line Votes year if state=="DC"
 
 *Electoral votes computation for 2020 election
gen realev2020=.
replace realev2020=EV2020 if Votes>0.5
total realev2020
 
gen estev2020=.
replace estev2020=EV2020 if vphat>0.5
total estev2020
 
list state realev2020 estev2020 uhat if year==2020
 
*Electoral votes computation for 2012 election


*OUR FULL MODEL PROPOSAL
reg Votes G_i P_i Z_i DPER DUR i tradition popdensity unemployment PerIncome Militexp TurnOut i.n_state if year<2020, robust
predict vphat3 if year==2020, xb
predict uhat3 if year==2020, resid

gen estev2020_2=.
replace estev2020_2=EV2020 if vphat3>0.5
total estev2020_2

list state realev2020 estev2020_2 Votes vphat3 uhat3 if year==2020

*ADJUSTMENTS TO OUR MODEL PROPOSAL*
gen U_i = unemployment*i
gen M_i = Militexp*i
gen GU = G*unemployment
gen PU = PerIncome*unemployment

reg Votes G_i P_i Z_i DPER DUR i tradition popdensity U_i M_i GU TurnOut i.n_state if year<2020, robust
reg Votes G_i P_i Z_i DPER DUR i tradition popdensity U_i M_i TurnOut i.n_state if year<2020, robust
reg Votes G_i Z_i DPER DUR i tradition popdensity U_i M_i i.n_state if year<2020, robust
reg Votes G_i Z_i DPER DUR tradition popdensity U_i M_i i.n_state if year<2020, robust

predict vphat_model2 if year==2020, xb
predict uhat_model2 if year==2020, resid

gen estev2020_model2=.
replace estev2020_model2=EV2020 if vphat_model2>0.5
total estev2020_model2

list state realev2020 estev2020_model2 Votes vphat_model2 if year==2020

****NAIVE LASSO ATTEMPT****
*We do not include G_i as we want it to be in the model (and not to get penalized by the Lasso)
lasso linear Votes P_i Z_i DPER DUR i i.n_state tradition popdensity unemployment PerIncome Militexp TurnOut if year<2020
est sto lasso_naive

predict vphat4 if year==2020, xb
*NOTE: we don't need to compute uhat for the Lasso *Ask Luca*
*predict uhat4 if year==2020, resid
gen estev2020_l=.
replace estev2020_l=EV2020 if vphat4>0.5
total estev2020_l

*Compare the first prediction of 2020 elections with the real output and the Naive Lasso approach
list state realev2020 estev2020 estev2020_l Votes vphat vphat4 if year==2020
graph bar vphat vphat4 if year==2020, over (state)
***GRAPH MUST BE ADEQUATELY GENERATED (MAYBE ANOTHER GRAPH IS BETTER)***

*adjust this command since its wrong
*list state realev2020 estev2020_l uhat if year==2020


****DOUBLE STAGE LASSO ATTEMPT (CROSS VALIDATION - DEFAULT METHOD)****

*Create globals of variables
global X_lasso P_i Z_i DPER DUR i tradition popdensity U_i M_i GU TurnOut
global X_effects D2-D51

*First Stage: lasso G_i (independent variable) vs controls
lasso linear G_i ($X_effects) $X_lasso if year<2020
est sto lasso1
lassocoef
*According to this output, we must include P_i Z_i DPER DUR i tradition popdensity U_i M_i GU TurnOut*
global X_depvar P_i Z_i DPER DUR i tradition popdensity U_i M_i GU TurnOut

*Second Stage: lasso Votes (dependent variable) vs penalized independent variables
lasso linear Votes ($X_effects) $X_lasso if year<2020
est sto lasso2
lassocoef
*In this case, the second stage Lasso selected Z_i DPER DUR tradition popdensity U_i M_i GU TurnOut
global X_second Z_i DPER DUR tradition popdensity U_i M_i GU TurnOut

*Third stage: Regress Votes on G_i
reg Votes G_i $X_lasso $X_second $X_effects if year<2020, robust
est sto lasso_ds
estimates table reg_2020 lasso_ds

predict vphat_lds if year==2020, xb
predict uhat_lds if year==2020, resid

gen estev2020_lds=.
replace estev2020_lds=EV2020 if vphat_lds>0.5
total estev2020_lds

list state Votes vphat vphat_lds if year==2020
graph bar Votes vphat vphat_lds if year==2020, over (state)
**ADJUST GRAPH IF WANT TO USE IN THE REPORT***

list realev2020 estev2020 estev2020_2 estev2020_lds if year==2020


****DOUBLE STAGE LASSO ATTEMPT (HETEROSKEDASTICITY...)****

*Create globals of variables
global X_lasso_H P_i Z_i DPER DUR i tradition popdensity U_i M_i GU TurnOut

*First Stage: lasso G_i (independent variable) vs controls
lasso linear G_i ($X_effects) $X_lasso_H if year<2020, selection(plugin, heteroskedastic)
est sto lassoH
lassocoef
*According to this output, we must include Z_i DPER U_i M_i*
global X_depvar_H Z_i DPER U_i M_i

*Second Stage: lasso Votes (dependent variable) vs penalized independent variables
lasso linear Votes ($X_effects) $X_lasso_H if year<2020, selection(plugin, heteroskedastic)
est sto lasso2
lassocoef
*In this case, the second stage Lasso selected Z_i DUR U_i*
global X_second_H Z_i DUR U_i

*Third stage: Regress Votes on G_i
reg Votes G_i $X_depvar_H $X_second_H $X_effects if year<2020, robust
est sto lasso_H
estimates table reg_2020 lasso_H

predict vphat_lH if year==2020, xb
predict uhat_lH if year==2020, resid

gen estev2020_lH=.
replace estev2020_lH=EV2020 if vphat_lH>0.5
total estev2020_lH

list state Votes realev2020 estev2020 estev2020_2 estev2020_lds estev2020_lH if year==2020

****DOUBLE STAGE LASSO ATTEMPT (ADAPTIVE SELECTION)****

*Create globals of variables
global X_lasso_AS P_i Z_i DPER DUR i tradition popdensity U_i M_i GU TurnOut

*First Stage: lasso G_i (independent variable) vs controls
lasso linear G_i ($X_effects) $X_lasso_AS if year<2020, selection(adaptive)
est sto lassoAS
lassocoef
*According to this output, we must include P_i Z_i DPER DUR tradition popdensity U_i M_i TurnOut*
global X_depvar_AS P_i Z_i DPER DUR tradition popdensity U_i M_i TurnOut

*Second Stage: lasso Votes (dependent variable) vs penalized independent variables
lasso linear Votes ($X_effects) $X_lasso_AS if year<2020, selection(adaptive)
est sto lasso2
lassocoef
*In this case, the second stage Lasso selected Z_i DPER DUR tradition popdensity U_i M_i*
global X_second_AS P_i Z_i DPER DUR tradition popdensity PerIncome unemployment Militexp TurnOut

*Third stage: Regress Votes on G_i
reg Votes G_i $X_depvar_AS $X_second_AS $X_effects if year<2020, robust
est sto lasso_AS
estimates table reg_2020 lasso_AS

predict vphat_lAS if year==2020, xb
predict uhat_lAS if year==2020, resid

gen estev2020_lAS=.
replace estev2020_lAS=EV2020 if vphat_lAS>0.5
total estev2020_lAS

list state Votes realev2020 estev2020 estev2020_2 estev2020_lds estev2020_lH estev2020_lAS if year==2020



**2012 PREDICTIONS***
*Fair´s Model
reg Votes G_i P_i Z_i DPER DUR i i.n_state if year<2012, robust

*Endogeneity Test 2012
*xtreg Votes G_i P_i Z_i DPER DUR i if year<2012, fe robust

predict vphat2 if year==2012, xb
predict uhat2 if year==2012, resid
list state realev2020 estev2020 Votes vphat2 uhat2 if year==2012

gen realev2012=.
replace realev2012=EV2012 if Votes>0.5
total realev2012
 
gen estev2012=.
replace estev2012=EV2012 if vphat2>0.5
total estev2012

list state realev2012 estev2012 Votes vphat2 uhat2 if year==2012

*MODEL PROPOSAL*
reg Votes G_i P_i Z_i DPER DUR i tradition popdensity unemployment PerIncome Militexp TurnOut i.n_state if year<2012, robust
predict vphat2012_2 if year==2012, xb
predict uhat2012_2  if year==2012, resid

gen estev2012_2=.
replace estev2012_2=EV2012 if vphat2012_2 >0.5
total estev2012_2

list state realev2012 estev2012 estev2012_2 Votes vphat2012_2  uhat2012_2  if year==2012



****DOUBLE STAGE LASSO ATTEMPT (CROSS VALIDATION - DEFAULT METHOD) 2012****

*Create globals of variables
global X_lasso_2012 P_i Z_i DPER DUR i tradition popdensity U_i M_i GU TurnOut
global X_effects_2012 D2-D51

*First Stage: lasso G_i (independent variable) vs controls
lasso linear G_i ($X_effects) $X_lasso_2012 if year<2012
est sto lasso1
lassocoef
*According to this output, we must include P_i Z_i DPER DUR tradition U_i M_i*
global X_depvar_2012 P_i Z_i DPER DUR tradition U_i M_i

*Second Stage: lasso Votes (dependent variable) vs penalized independent variables
lasso linear Votes ($X_effects) $X_lasso_2012 if year<2012
est sto lasso2
lassocoef
*In this case, the second stage Lasso selected P_i Z_i DPER DUR i tradition U_i M_i GU TurnOut
global X_second_2012 P_i Z_i DPER DUR i tradition U_i M_i GU TurnOut

*Third stage: Regress Votes on G_i
reg Votes G_i $X_lasso_2012 $X_second_2012 $X_effects if year<2012, robust
est sto lasso_ds
estimates table reg_2012 lasso_ds

predict vphat_lds_2012 if year==2012, xb
predict uhat_lds_2012 if year==2012, resid

gen estev2012_lds=.
replace estev2012_lds=EV2012 if vphat_lds_2012 >0.5
total estev2012_lds

****DOUBLE STAGE LASSO ATTEMPT (HETEROSKEDASTICITY) 2012****

*Create globals of variables
global X_lasso_2012_H P_i Z_i DPER DUR i tradition popdensity U_i M_i GU TurnOut

*First Stage: lasso G_i (independent variable) vs controls
lasso linear G_i ($X_effects) $X_lasso_2012_H  if year<2012, selection(plugin, heteroskedastic)
est sto lasso1
lassocoef
*According to this output, we must include Z_i DPER U_i M_i*
global X_depvar_2012_H Z_i DPER U_i M_i

*Second Stage: lasso Votes (dependent variable) vs penalized independent variables
lasso linear Votes ($X_effects) $X_lasso_2012_H if year<2012, selection(plugin, heteroskedastic)
est sto lasso2
lassocoef
*In this case, the second stage Lasso selected Z_i DUR U_i
global X_second_2012_H Z_i DUR U_i

*Third stage: Regress Votes on G_i
reg Votes G_i $X_lasso_2012_H $X_second_2012_H $X_effects if year<2012, robust
est sto lasso_ds
estimates table reg_2012 lasso_ds

predict vphat_lds_2012_H if year==2012, xb
predict uhat_lds_2012_H if year==2012, resid

gen estev2012_lds_H=.
replace estev2012_lds_H=EV2012 if vphat_lds_2012_H >0.5
total estev2012_lds_H

list state Votes realev2012 estev2012 estev2012_2 estev2012_lds estev2012_lds_H if year==2012