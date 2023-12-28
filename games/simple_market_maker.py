import random
import statistics

init_cash = 10000

gains_by_day = []
for days in range(250):  # 250 days of trading
    # our trader gets a fixed budget to trade with each day
    cash = init_cash

    # we start each day owning nothing
    stock = 0

    # initial value of the stock always starts at $100/share
    value = 100

    # buy-price and sell-price are always set one cent away from last trade
    gap = 1

    buy_at = value - gap
    sell_for = value + gap
    for trades in range(25000):  # 25K trades per day
        if cash < 0:
            break  # lost all the money - done for the day

        customer_thinks = random.gauss(value, 10*gap)

        if customer_thinks > sell_for:  # it's cheap! - they will buy from you
            #print(f"We sell for {sell_for}")
            stock -= 1
            cash += sell_for

            value = sell_for  # raise the current estimated value to the sell price

            # now we are (net) down a share and are more eager to buy
            buy_at = value - gap
            sell_for = value + gap

        elif customer_thinks < buy_at:  # you're a fool! - they will sell to you
            #print(f"We buy at {buy_at}")
            stock += 1
            cash -= buy_at
            value = buy_at  # lower the current estimated value to the buy price

            # now we are (net) up a share and are more eager to sell
            # so we lower the price that we will sell at
            buy_at = value - gap
            sell_for = value + gap

        else:
            # no trade happens
            None

        #buy_at = value - gap
        #sell_for = value + gap

        if sell_for < buy_at:
            print("INVERTED!")

        #print(f"Value: {value} \t Bid/Ask: {buy_at}/{sell_for} \t Cash: {cash} \t Stock: {stock}")

    # at the end of the day we have to get back to flat:
    cash += stock * value  # sell all the shares (or buy back if we're short)
    gains_by_day.append((cash - init_cash)*1.0/init_cash)

print(gains_by_day)

average_gain = statistics.mean(gains_by_day)
std_of_gain = statistics.stdev(gains_by_day)

print(f"Average daily gain: {100*average_gain:.2f}%")
print(f"Std Dev per day: {100*std_of_gain:.2f}%")
#print(f"Sharpe Ratio: {16*average_gain/std_of_gain}")