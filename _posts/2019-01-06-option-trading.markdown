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

($$\delta$$) The rate of change in an options' value with respect to movement in the price of the underlying contract.

Calls have positive delta values between 0 and 1.00 (100) <br/>
Puts have negative delta values between 0 and -1.00 (-100) <br/>

Delta is also approximately the <span  style="background-color: #FFFF00"> probability that an option will finish in-the money</span>. E>g. Delta = 10, 10% chane of finishing in-the-money

Eg. Option delta = 50 (0.5). If underlying price up 1.20, option value up 0.6. If underlying price down 1.70, option price down 0.85

#### Gamma (Curvature)

($$\gamma$$ Change in delta) The rate of change in an option's delta respect to movement in the price of the underlying contract

<span  style="background-color: #FFFF00">  All options have positive gamma values </span>

