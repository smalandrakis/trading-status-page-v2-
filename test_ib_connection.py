"""
Test IB Gateway Connection

Simple script to test connection to IB Gateway on port 4002
and retrieve BTC futures contract details
"""

from ib_insync import IB, Future, util
import asyncio

async def test_connection():
    ib = IB()

    print("="*60)
    print("Testing IB Gateway Connection (Port 4002)")
    print("="*60)

    try:
        # Connect to paper trading port
        await ib.connectAsync('127.0.0.1', 4002, clientId=1)
        print("✓ Connected to IB Gateway")

        # Get account details
        account = ib.managedAccounts()[0]
        print(f"✓ Account: {account}")

        # Define BTC futures contract
        # Using MBT (Micro Bitcoin Futures)
        contract = Future(
            symbol='MBT',
            lastTradeDateOrContractMonth='202409',  # Update to current front month
            exchange='CME',
            currency='USD'
        )

        # Qualify the contract
        contracts = await ib.qualifyContractsAsync(contract)

        if contracts:
            print(f"✓ Contract qualified: {contracts[0]}")

            # Request market data
            ib.reqMktData(contracts[0], '', False, False)
            await asyncio.sleep(2)

            # Get ticker data
            ticker = ib.ticker(contracts[0])

            print(f"\nMarket Data:")
            print(f"  Last: ${ticker.last:,.2f}" if ticker.last else "  Last: N/A")
            print(f"  Bid:  ${ticker.bid:,.2f}" if ticker.bid else "  Bid: N/A")
            print(f"  Ask:  ${ticker.ask:,.2f}" if ticker.ask else "  Ask: N/A")
            print(f"  Close: ${ticker.close:,.2f}" if ticker.close else "  Close: N/A")

            # Get current positions
            positions = ib.positions()
            if positions:
                print(f"\nCurrent Positions:")
                for pos in positions:
                    print(f"  {pos.contract.symbol}: {pos.position} @ ${pos.avgCost:,.2f}")
            else:
                print("\n✓ No current positions")

            # Get account values
            account_values = ib.accountValues()
            for val in account_values:
                if val.tag == 'NetLiquidation':
                    print(f"\nAccount Net Liquidation: ${float(val.value):,.2f}")
                    break

        else:
            print("❌ Failed to qualify contract")
            print("\nPossible issues:")
            print("  - Check contract month (update to current front month)")
            print("  - Verify BTC futures permissions in IB account")
            print("  - Try 'BTC' instead of 'MBT' for standard contract")

        ib.disconnect()
        print("\n✓ Disconnected")

    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Is IB Gateway running?")
        print("  2. Is it configured for paper trading (port 4002)?")
        print("  3. Is API access enabled in Gateway settings?")
        print("  4. Is the Gateway logged in?")


if __name__ == '__main__':
    asyncio.run(test_connection())
