---
layout:     post
title:      "Option Trading"
subtitle:   "Note"
date:       2019-01-06 12:00:00
author:     "Becks"
header-img: "img/post-bg-city-night.jpg"
catalog:    true
tags:
    - 总结
    - 学习笔记
---

> note from Sheldon Natenberg

<script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script>


## Basic Concept

![](\img\post\Deep-Learning\pic3.png)

**forward contract**: A forward contract is an agreement between a buyer and seller whereby the price is negotiated right now, but the actual exchange of money for goods takes place at a later date (the <span style="color: red">maturity date, delivery date, or expiration date</span>).

The **nominal value or notional value** of a physical <span style="background-color: #FFFF00">commodity futures contract </span>: the unit value multiplied by the number of units to be delivered. unit value = $1175。 units to be delivered = 100 nominal value = $1175 x 100 = $117,500

Contract month codes: <br/>
F - January  <br/>
G - February H - March <br/>
J - April  <br/>
K - May M - June <br/>
N - July  <br/>
Q - August  <br/>
U - September  <br/>
V - October  <br/>
X - November <br/> 
Z - December  

The **nominal value or notional value** of an <span style="background-color: #FFFF00">index or cash-settled futures contract </span>: the index value multiplied by the point value, or index multiplier (SPZ6) index value = 2200. point value = $250 nominal value = 2200 x $250 = $550,000


Futures-type Settlement: 

Buy 1 S&P futures contract at 2200 total value of trade = 2200 x $250 = $525,000 payment from buyer to seller = 0. **margin deposit** with clearing house 

Futures contract rises to 2210 profit = 10 x +$250 = +$2,500 <br/>
Futures contract falls to 2190 loss = 10 x -$250 = -$2,500 <br/>
<span style="background-color: #FFFF00">**Variation**</span> – the daily realized profit or loss on an open futures position <br/>


Stock is always subject to stock-type settlement <br/>
Futures are always subject to futures-type settlement (sometimes referred to as **margin and variation**) <br/>
Options may be subject to either stock-type or futures-type settlement. <br/>

On most option markets around the world options are subject to the same settlement procedure as the underlying contract –

If the underlying for the option is stock (or a security), the options are subject to stock-type settlement.<br/>
If the underlying for the option is a futures contract, the options are subject to futures-type settlement.<br/>


<span style="color: red">In the United States all options, whether options on stock or options on futures, are subject to stock-type settlement. All options must be paid for fully in cash. </span>

In the United States: <br/>
Stock – three business days (T+3)<br/>
Options – the next business day (T+1) <br/>
Futures – immediate (T+0)

**Clearing House** – the organization responsible for processing and guaranteeing all trades madeonanexchange. Theclearinghouse may be a division of the exchange, or a completely separate and independent entity.

Some major clearing houses: <br/>
Depository Trust & Clearing Corp. (DTCC) New York Stock Exchange <br/>
Options Clearing Corp. (OCC) All U.S. stock option exchanges   <br/>

#### Dividend Dates

<span style="background-color: #FFFF00">**Declared date**</span> – the date on which the company announces the amount of the dividend and the date on which it will be paid. 

<span style="background-color: #FFFF00">**Record date**</span> – the date on which the stock must be owned in order to receivethedividend. Ownership of the stock does not become official until the settlement date.

In the United States stock normally settles on the third business day following the transaction.
trade date (T) settlement date (T+3). <span style="color: red">In order to receive the dividend the stock must be purchased three business days prior to the record date. (R-3)</span>

<span style="background-color: #FFFF00">**Ex-dividend date (ex-date)**</span> – the first day on which the stock is trading without the rights to the dividend. Because
stock settles in three business days, the ex-date is two days prior to the record date. (R-2)


**Payable date** – the date on which the dividend will be paid to qualifying shareholders (those owning stock on the record date).


<span style="background-color: #FFFF00">**Short Sale**</span> The sale by a trader of stock which the trader does not own. In order to sell stock short a trader must borrow the stock. Before a trader can sell stock short, he must confirm that the stock can be borrowed (locate). If a stock is unavailable for borrowing, the trader will usually be informed of this by his clearing firm.

The short seller must pay the lender any dividends which accrue. <br/>
The short seller will have to pay the lender some amount for the privilege of borrowing the stock.<br/>
This amount is usually expressed as a percentage rate. <br/>

**borrowing costs ($$r_{bc}$$)**: percent interest paid to the lender of stock


**long rate ($$r_{l}$$)**: percent interest earned on the sale of long stock <br/>
**shortrate($$r_{s}$$) (short stock rebate)**: percent interest earned on the short sale of stock
$$r_{bc} = r_{l} - r_{s} $$ <br/>
The more difficult a stock is to borrow, the higher the borrowing costs. For very difficult to borrow stock the short seller may earn no interest. $$ r_{s} = 0, r_{bc}= rl$$


When subject to stock-type settlement, options, whether bought or sold, are always subject to the long rate. <br/>
If we buy options, we pay the long rate.<br/> 
If we sell options, we earn the long rate.


#### Option Contract 

<span style="background-color: #FFFF00">**Expiration Date (Expiry)** </span>– The date on which the buyer of an option must make a final decision whether to buy, in the case of a call, or sell, in the case of a put.

U.S. stock and stock index markets <br/>
– the third Friday of the expiration month.<br/>
Foreign stock option markets <br/>
– usually the same as U.S. markets, but may vary by exchange<br/>
Futures options on physical commodities <br/>
– usually the month prior to the contract month; the exact date determined by each exchange


<span style="background-color: #FFFF00">**European** </span> – the option may be exercised only at expiration. All stock index options are European

<span style="background-color: #FFFF00">**American** </span> – the option may be exercised at any time prior to expiration. Most individual equity options, as well as most options on futures, are American


#### Settlement 

<span style="background-color: #FFFF00">**Settlement into a future position** </span>– 
1. Exercise results in a future position 
2. The position is immediately subject to margin and variation
3. There is a credit or debit equal to the difference between the exercise price and current futures price

E.g. March S&P 500 futures = 2150, exerice one S&P 500 March 2100 call. You become long 1 March S&P 5000 futures contract at a price at a price of 2100. You must deposit the appropriate margin. Your account is credited with (2150-2100)\*250 = $12500


<span style="background-color: #FFFF00">**Settlement into a cash position** </span>– 
1. Used primarily for options on a cash index or for options on futures when the optionand underlying futures contract expire at the same time.
2. Exerice results in no underlying position
3. There is a cash credit or debit equal to the difference between the exercise price and current underlying price

E.g. Stick index = 525. Each index point has a value of $100, Exereice one 475 call. Account is credited with (525 -475)\*$100 = $ 5000


#### Volatility 

If we assume **normal distribution** of prices. One s.d. range $$F * \sigma * \sqrt t$$

If we assume **lognormal distribution** of prices. One standard deviation: $$ F * e^{n * \sigma * \sqrt t}$$

1.  In total points an <span style="background-color: #FFFF00">**at-the-money** </span> option is always more sensitive to a change in volatility than an equivalent in- or out-of-the-money option (波动性改变对 at the money total points 改变最大)
2. In percentage terms an <span style="background-color: #FFFF00">**out-of-the-money** </span> option is always more sensitive to a change in volatility than an equivalent in- or at-the-the-money option  (波动性改变的让 out of the money percentage变化最大）
3. A long-term option is always more sensitive to a change in volatility than an equivalent short-term option (长期比短期个更对波动性敏感)

注: For <span style="color: red">short-term</span> interest rate products, volatility calculations are alwasy made using the inerest rate associated with the contract, not the contract price iteself。 例如: Euodollars, Euribor, Short Sterling. For <span style="color: red">Long-term</span> interest rate products such as trasury bonds and notes, volatility calculations are made using the actual contract price. 例如 Eurollar price = 95.75, annual volatilty of Eurodollars futures is 32%, daily standard deviation = (100-95.75) \* 32% /16 = 0.09

Example. Eurodollar futures = 96.72, Volatility = 29.56%, Time to expiration = 77 days <br/>
Assuming a lognormal distribution of interest rates, what is the likelihood that the 97.50 call will be in-the-money at expiration?<br/>
Interest rates are currently 100 - 96.72 = 3.28(%). <br/>
For the 97.50 call to be in-the-money at expiration, interest rates must be below 100 - 97.50 = 2.50(%) <br/>
One standard deviation = .2956\*√(77/365) = .1358 Number of standard deviations: ln(2.50/3.28)/.1358 = 2.00 A 97.50 call is equivalent to a 2.50% put.


## Greeks

#### Delta

($$\Delta$$) The rate of change in an options' value with respect to movement in the price of the underlying contract.

Calls have positive delta values between 0 and 1.00 (100) <br/>
Puts have negative delta values between 0 and -1.00 (-100) <br/>

Delta is also approximately the <span  style="background-color: #FFFF00"> probability that an option will finish in-the money</span>. E>g. Delta = 10, 10% chane of finishing in-the-money

Eg. Option delta = 50 (0.5). If underlying price up 1.20, option value up 0.6. If underlying price down 1.70, option price down 0.85

注：buy calls and sell puts 是long delta position

#### Gamma (Curvature)

($$\Gamma$$ Change in delta) The rate of change in an option's delta respect to movement in the price of the underlying contract

<span  style="background-color: #FFFF00">  All options have positive gamma values </span>

At-the-money gamma 最大, 因为delta vs price change graph slope is greatest when close to options's exercise price.

#### Theta 

($$\theta $$) The sensitivity of an option's value to the passage of time.

Theta depends on two factors:  <br/>
decay in <span style="color: red">volatility value </span> <br/>
decay in <span style="color: red"> interest value </span>  <br/>
    
Since volatility value is usually more important than interest, the great majority of options lose value as time passes. 
注: An option which loses value as time passes will have a negative theta （theta 一般是负的）

Q： What would a positive theta mean? As time passes the option becomes more valuable. Is this possible? （interest rate 比 volatility 考虑的多）

The expected value of the option at expiration must be very close to intrinsic value. value today = present value of the expected value. If there are no changes in other market conditions, as time passes, the value of the option will rise
to intrinsic value. 比如underlying = 100, call = 40, 其他的不变，present value of 60: 60/(1+r\*t). As time passes, the option value will rise to 60


#### Vega

($$\Vega$$) The senstivity of an options's value to a change in volatility 

Usually expressed as the change in value **per one percentage point change** in volatility

The vega is often interpreted as the sensitivity of an option’s price to a change in <span  style="background-color: #FFFF00">implied volatility</span>.

<span  style="background-color: #FFFF00"> All options have positive vega values</span>: If we raise volatility, we raise the value of the option

#### Rho

($$\rho$$) The sensitivity of an option’s value to a change in interest rates

Usually expressed as the change in value **per one percent change** in interest rates

If the underlying is a futures contract, and options are subject to <span style="color: red"> futures-type settlement</span>, <span style="color: red">all options have a rho value of zero</span>. Changes in interest rates will have no effect on an option’s value.


or options on futures, where the options are subject to stock-type settlement as they are in the U.S., all options have negative rho values. When we raise interest rates, we <span style="color: red">reduce the present value of the option</span>.

If raising interest rates increases the forward price, as it does for stocks, then... <br/>
<span  style="background-color: #FFFF00">calls have positive rho values </span> <br/>
<span  style="background-color: #FFFF00">puts have negative rho values </span> <br/>

rho is the least important than delta, gamma, theta, vega

#### Dividend Risk

Raise dividends: lower forward price,  calls decrease, puts increase <br/>
Lower dividends: raise forward price. call increase, puts decrease<br/>

<span  style="background-color: #FFFF00">In stock option markets, interest and dividends always have the opposite effect on option values.</span>

Q. 怎么measure change dividend effect？ A: 用delta

E.g.  Dividend is raised 0.34, call delta = 70, call decrease 0.34 \* 0.7 = 0.24.  put delta = -20, put increase 0.34\*0.2 = 0.07. Dividend is cut 0.51, call delta = 35, call increase 0.51 \* 0.35 = 0.18

| measure | calls   |  puts  | underlying  |
| ------------ | ------------ | ------------ | ------------ |
| delta  | positive   | negative  | positive  |
| gamma  | positive   | positive   | zero  |
| theta | negative  | negative  | zero  |
| vega | positive | positive | zero  |
| rho (stocks) | positive | negative  | zero  |
| rho (futures) | negative  | negative  | zero |

| measure | In-the-money   |  at-the-money  | out-of-the-money  |
| ------------ | ------------ | ------------ | ------------ |
| gamma  |    | ✔️   |   |
| theta  |    | ✔️   |   |
| vega |   | ✔️   |   |

1. <span  style="background-color: #FFFF00">An **at-the-money option** always has a greater gamma, theta, and vega than an equivalent in-the-money or out-of-the-money option. </span> 
2. <span  style="background-color: #FFFF00">At-the-money options tend to be the most actively traded. </span>
3. <span  style="background-color: #FFFF00">时间越长，vega约敏感 A long-term option always has a greater vega value than an equivalent short-term option. </span> 比如40% upward to 10, 60% downward to 10, 只有5秒expire 和 还有6周expire 比，还有六周的expire underlying更容易发生变化
4. <span  style="background-color: #FFFF00"> 约靠近expiration, ATM theta 增加 </span> As time passes the theta of an at-the-money option increases.
5. <span  style="background-color: #FFFF00"> in-the-money, rho大, An in-the-money option has a greater rho value than an equivalent at-the-money or out-of-the-money option </span>
6. <span  style="background-color: #FFFF00"> 时间越长，rho越大，Along-term option has a greater rho value than equivalent short-term option.</span>





##  Spread

A spread, usually delta neutral, which is sensitive to either the volatility of the underlying contract (**gamma**), or to changes in implied volatility (**vega**)

#### Straddle

Long Straddle: +1 June 100 call +1 June 100 put ; <br/>
Short Straddle: -1 June 100 call -1 June 100 put   <br/>

#### Strangle

Long Strangle: +1 June 105 call +1 June 95 put ; <br/>
Short Strangle: -1 June 105 call -1 June 95 put   <br/>

#### Bufferfly

Long Bufferfly: +1 July 95 call (wing), -2 July 100 calls (body), +1 July 105 call (wing）; +1 August 90 put (wing) -2 August 100 puts (body) +1 August 110 put (wing) <br/>
Short Bufferfly: -1 July 95 call  (wing), +2 July 100 calls (body), -1 July 105 call(wing）; -1 August 90 put +2 August 100 puts -1 August 110 put   <br/>

![](\img\post\option-trading\spread.png)

#### Ratio Spread

underlying price = 100 <br/>
Buy more than Sell: +3 August 105 call (delta 25) -1 August 95 call (delta 75);  +2 September 95 put (delta: -25)
-1 September 100 put (delta: -50) <br/>
Sell more than Buy: -3 August 105 call +1 August 95 call <br/>

![](\img\post\option-trading\ratio-spread.png)

#### Calendar Spread

also called Time Spread, Horizontal Spread

Long Calendar Spread: +1 September 100 call  -1 July 100 call ; +1 November 65 put -1 October 65 put <br/>
Short Calendar Spread: -1 September 100 call +1 July 100 call; -1 November 65 put +1 October 65 put <br/>

![](\img\post\option-trading\long-calendar-spread.png)

![](\img\post\option-trading\short-calendar-spread.png)


#### Bull and Bear Spread

Bull (Vertical) Spread Buy an option at a lower exercise price Sell an option at a higher exercise price <br/> 
+1 December 100 call -1 December 110 call; Or +1 December 100 put -1 December 110 put; <br/>
minimum value = 0 maximum value = Xh - Xl

Bear (Vertical) Spread Buy an option at a higher exercise price Sell an option at a lower exercise price <br/> 
-1 December 100 call +1 December 110 call Or -1 December 100 put +1 December 110 put <br/>
minimum value = 0 maximum value = Xh - Xl

Both options must be the same type (both calls or both puts) and expire at the same time.



![](\img\post\option-trading\bull-spread.png)

![](\img\post\option-trading\bear-spread.png)

| spread | delta | gamma | theta  | vega |
| ------------ | ------------ | ------------ | ------------ | ------------ | 
| Long Straddle/Strangle | 0  |  +  | -  |  +  |
| Short Straddle/Strangle  |  0  | -  |  + | - |
| Long Butterfly | 0  |  -  | +  |  -  |
| Short Butterfly  |  0  | +  |  - | + |
| Ratio Spread (Buy more than sell)| 0  |  + | -  | +  |
| Ratio Spread (Sell more than Buy)  |  0  | -  |  + | - |
| Long Calendar Spread | 0  |  -  | +  |  +  |
| Short Calendar Spread   |  0  | +  |  - | -- |


| spread | Downside Risk / Reward  | Upside Risk / Reward |
| ------------ | ------------ | ------------ |
| Long Straddle / Strangle | unlimited reward | unlimited reward |
| Short Straddle / Strangle  | unlimited risk  | unlimited reward |
| Long Butterfly  |  limited risk |  limited risk |
| Short Butterfly  | limited reward | limited risk |
| Call Ratio Spread (Buy more than sell)| limited reward | unlimited reward  |
| Put Ratio Spread (Buy more than sell)| unlimited reward | limited reward  |
| Call Ratio Spread (Sell more than Buy)  |  limited risk | unlimited risk |
| Put Ratio Spread (Sell more than Buy)  |  unlimited risk | limited risk |

Interest Rate Increase:  (时间越长，rho 越大)

Long Call Calendar Spread Rho ⬆️,  +1 September 1 call rho ⬆️⬆️, -1 July 100 call ⬆️ <br/>
Long Put Calendar Spread  ⬇️ +1 September 100 put ⬇️⬇️  -1 July 100 put ⬇️ <br/>

Dividend (跟rho 相反)

Long Call Calendar Spread Rho ⬇️ ,  +1 September 1 call   -1 July 100 call
Long Put Calendar Spread ⬆️, +1 September 100 put  -1 July 100 put 

| gamma / vega | spread|
| ------------ | ------------ |
| +   + | longstraddle, longstrangle, short butterfly, ratio spread (buy more than sell) | 
| - - | short straddle, short strangle, long butterfly, ratio spread (sell more than buy) | 
| - + | long calendar spread |
| + - | short calendar spread |

#### Decision

Q: Under what conditions might you choose a difference spread: <br/>
A: Most trading decisions depend on price vs. value. If something has a high price and a low value,prefer to be a seller. If something has a low price and a high value, prefer to be a buyer. In option trading.... <br/>
<span  style="background-color: #FFFF00"> price = implied volatility value = (future) realized volatility </span>

If you believe the future volatility over the life of the option(s) will be higher than the current implied volatility, you want to be a buyer of realized volatility. e.g. future realized volatility 25% implied volatility 25% . You want to create a position with a positive gamma: long straddles, long strangles, short butterflies, ratio spreads where you buy more than sell

If you believe the future volatility over the life of the option(s) will be lower than the current implied volatility, you want to be a seller of realized volatility. e.g. future realized volatility 20%  implied volatility 25%
You want to create a position with a negative gamma: short straddles short strangles long butterflies ratio spreads where you sell more than buy

If you believe <span  style="background-color: #FFFF00">  implied volatility will rise at least as quickly as realized volatility </span>, you want to create a position with a positive vega:  buy calendar spreads

If you believe <span  style="background-color: #FFFF00">  implied volatility will fall at least as quickly as realized volatility </span>, you want to create a position with a negative vega: sell calendar spreads

## Synthetics

long call + short put = sythetic long underlying ≈ long underlying

short call + long put = sythetic short underlying ≈ short underlying

Both options must have the same exercise price and expiration date.

long put + long underlying ≈ sythetic long call <br/>
short put + short underlying ≈ sythetic short call  <br/>
long call + short underlying ≈ sythetic long put <br/>
short call + long underlying ≈ sythetic short put  <br/>

Any strategy can be done using a synthetic equivalent: 

e.g. 
+2 December 100 calls -1 underlying contract; <br/>
+1 December 100 call, combine +1 December 100 call and  -1 underlying contract <br/>
+1 December 100 call +1 December 100 put <span style="color: red"> Straddle </span>

## Put Call Parity

$$C - P = \left( F - X \right) / \left(1 + r\*t\right)$$




