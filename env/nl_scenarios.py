"""
nl_scenarios.py

The natural-language scenario bank for NLConditionalEnv, split out of
env/nl_conditional_env.py because it outgrew the module that uses it.

Why 60 and not 12
-----------------
With a deterministic (greedy / argmax) policy and no per-episode randomness at
alpha=0, every repeat of a scenario produces a byte-identical trajectory. The
12-scenario bank therefore gave n_effective = 12 no matter how many episodes
were requested: all four models' alpha=0 rates came out as exact multiples of
1/12 (0.333 = 4/12, 0.833 = 10/12, 0.083 = 1/12). Reporting those as n=48
overstates precision by ~2x. This is the same failure the 7-scenario bank had
(design requirement 4 in RESEARCH_PLAN.md); it was fixed once and reappeared.

60 scenarios puts n_effective at 60 and makes the metric granularity 1/60.

Invariants (enforced by validate_bank, which runs at import)
------------------------------------------------------------
  probe not in static_gold   probing a static task must be pure waste, or
                             over-gathering is unmeasurable by construction
  branch[0] != branch[1]     the latent bit must actually change the answer
  probe not in branch        the probe must not also be its own follow-up
  |static_gold| == 2         fixed cardinality, so top-k baselines are fair
  cond_query != static_query the query must signal WHETHER to probe
  result[0] != result[1]     the probe must be informative

The conditional cue is deliberately NOT one lexical marker
----------------------------------------------------------
If every cond_query contained "depending on", both the LLM and a linear probe
could separate the two classes on a single token, and "the query signals
whether the task is conditional" would be a statement about one keyword rather
than about meaning. The phrasings below are varied on purpose -- "as
appropriate", "whatever X calls for", "act on what Y shows", "handle it per
Z", "take the right step for", and several with no marker at all that are
simply underspecified. Static queries vary in length too, so sentence length is
not a usable shortcut either. experiments/check_scenario_bank.py measures how
much of the separability survives when the obvious markers are stripped.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class Scenario:
    cond_query: str                 # phrased so the right action depends on a check
    static_query: str               # fully specified; no check needed
    probe: str                      # tool that reveals the latent bit
    branch: Tuple[str, str]         # follow-up for bit=0 / bit=1
    result: Tuple[str, str]         # probe's textual result for bit=0 / bit=1
    static_gold: Tuple[str, str]    # gold for static episodes; must EXCLUDE probe


TOOLS: Dict[str, str] = {
    "WeatherTool":   "Get current weather conditions and any active alerts for a location.",
    "NewsTool":      "Search recent news articles on a topic.",
    "TripTool":      "Plan a trip itinerary, including transport and activities.",
    "HotelTool":     "Search and book hotel rooms.",
    "CarRentalTool": "Reserve a rental car.",
    "FlightTool":    "Check flight status and delays for a given flight.",
    "FinanceTool":   "Look up stock prices, market data, and account balances.",
    "TaxTool":       "Compute tax liability and file-related figures.",
    "BudgetTool":    "Create or adjust a personal budget.",
    "InvoiceTool":   "Issue or look up an invoice.",
    "HealthTool":    "Check symptoms and get general health information.",
    "PharmacyTool":  "Find nearby pharmacies and medication availability.",
    "DoctorTool":    "Book an appointment with a physician.",
    "InsuranceTool": "Check insurance coverage and claims.",
    "InventoryTool": "Check stock levels for a product.",
    "OrderTool":     "Place an order for a product.",
    "SupplierTool":  "Request a restock from a supplier.",
    "ShippingTool":  "Arrange shipment and track a parcel.",
    "CalendarTool":  "Check availability and events on a calendar.",
    "EmailTool":     "Send an email.",
    "MessageTool":   "Send an instant message.",
    "MeetingTool":   "Create a meeting with a video link.",
    "RepoTool":      "Inspect a code repository's build and test status.",
    "DeployTool":    "Deploy the current build to production.",
    "RollbackTool":  "Roll back production to the previous build.",
    "TicketTool":    "File or update an issue ticket.",
    "BillingTool":   "Look up a customer's billing record.",
    "RefundTool":    "Issue a refund to a customer.",
    "EscalateTool":  "Escalate a case to a human supervisor.",
    "AccountTool":   "Look up or modify account settings.",
    "SecurityTool":  "Check account security status and recent sign-ins.",
    "ResetTool":     "Reset a password or credential.",
    "LegalTool":     "Retrieve contract and compliance documents.",
    "HRTool":        "Look up employee records and leave balances.",
    "PayrollTool":   "Run or correct a payroll entry.",
    "TrainingTool":  "Enrol an employee in a training course.",
    "DONE":          "Stop: the tool set is complete.",
}

# The action space is intentionally UNCHANGED from the 12-scenario bank (36
# tools + DONE). Expanding the catalogue alongside the bank would have made the
# earlier LLM numbers ("DONE ranked 35th of 35") incomparable; tools are reused
# across scenarios instead, which is also the more realistic setting.

SCENARIOS: List[Scenario] = [
    # ---------------------------------------------------- travel / logistics
    Scenario(
        cond_query="I'm heading to Denver this weekend. Sort out whatever the conditions call for.",
        static_query="I'm going to Denver this weekend. Book me a hotel and a rental car.",
        probe="WeatherTool", branch=("TripTool", "NewsTool"),
        result=("Denver: clear skies, 22C, no alerts.",
                "Denver: severe winter storm warning in effect through Sunday."),
        static_gold=("HotelTool", "CarRentalTool"),
    ),
    Scenario(
        cond_query="I have flight UA 421 tonight. Make the arrangements that fit the situation.",
        static_query="For my flight UA 421 tonight, reserve a car and book a hotel at the destination.",
        probe="FlightTool", branch=("CarRentalTool", "HotelTool"),
        result=("UA 421 is on time, departing 21:05.",
                "UA 421 is cancelled; next available departure is tomorrow morning."),
        static_gold=("CarRentalTool", "HotelTool"),
    ),
    Scenario(
        cond_query="Figure out my Lisbon trip next week and handle it based on the forecast.",
        static_query="For Lisbon next week, book the flight and a hotel near the old town.",
        probe="WeatherTool", branch=("HotelTool", "TripTool"),
        result=("Lisbon: dry and mild all week, highs near 24C.",
                "Lisbon: heavy rain and coastal wind warnings midweek."),
        static_gold=("FlightTool", "HotelTool"),
    ),
    Scenario(
        cond_query="My connection in Frankfurt is tight. Do what the flight situation requires.",
        static_query="Plan my Frankfurt layover itinerary and put it on my calendar.",
        probe="FlightTool", branch=("HotelTool", "MessageTool"),
        result=("Inbound LH 1042 is early; the connection holds comfortably.",
                "Inbound LH 1042 is delayed 90 minutes; the connection will be missed."),
        static_gold=("TripTool", "CalendarTool"),
    ),
    Scenario(
        cond_query="Arrange the airport pickup for Tuesday as appropriate.",
        static_query="Put Tuesday's airport pickup on my calendar and email the driver the details.",
        probe="FlightTool", branch=("CarRentalTool", "MessageTool"),
        result=("Tuesday's arrival is on schedule at 14:20.",
                "Tuesday's arrival has moved to 23:50 after an aircraft swap."),
        static_gold=("CalendarTool", "EmailTool"),
    ),
    Scenario(
        cond_query="Sort out travel for the Oslo offsite, whatever the dates allow.",
        static_query="For the Oslo offsite, book flights and a hotel for three nights.",
        probe="CalendarTool", branch=("FlightTool", "MeetingTool"),
        result=("The team is free the week of the 14th; in-person travel works.",
                "Three of five are double-booked that week; in-person is not viable."),
        static_gold=("HotelTool", "FlightTool"),
    ),
    Scenario(
        cond_query="I may need to move my Rome hotel. Act on what the weather shows.",
        static_query="Plan my Rome itinerary and pull any recent news about transit strikes.",
        probe="WeatherTool", branch=("HotelTool", "CarRentalTool"),
        result=("Rome: settled and warm through the weekend.",
                "Rome: flooding reported in the district around the hotel."),
        static_gold=("TripTool", "NewsTool"),
    ),
    Scenario(
        cond_query="Rebook the Zurich leg if the situation calls for it.",
        static_query="For the Zurich leg, book a hotel and block the travel time on my calendar.",
        probe="FlightTool", branch=("TripTool", "EmailTool"),
        result=("The Zurich leg is confirmed and on time.",
                "The Zurich leg was cancelled by the carrier overnight."),
        static_gold=("HotelTool", "CalendarTool"),
    ),
    Scenario(
        cond_query="Confirm the Berlin room and handle whatever the booking shows.",
        static_query="Draft the Berlin trip plan and add the travel dates to my calendar.",
        probe="HotelTool", branch=("EmailTool", "CarRentalTool"),
        result=("The Berlin reservation is confirmed for four nights.",
                "The Berlin reservation was released; the property is now full."),
        static_gold=("TripTool", "CalendarTool"),
    ),
    Scenario(
        cond_query="Sort the car for the Munich leg as availability dictates.",
        static_query="For the Munich leg, book the flight and reserve a hotel room.",
        probe="CarRentalTool", branch=("TripTool", "MessageTool"),
        result=("A compact car is available at Munich airport for the full period.",
                "No vehicles are available in Munich for those dates."),
        static_gold=("HotelTool", "FlightTool"),
    ),
    Scenario(
        cond_query="Advise on Saturday's outdoor event, based on conditions.",
        static_query="Put Saturday's outdoor event on the calendar and email the attendees.",
        probe="WeatherTool", branch=("MessageTool", "TripTool"),
        result=("Saturday: sunny, light breeze, no alerts.",
                "Saturday: thunderstorm watch from midday onward."),
        static_gold=("CalendarTool", "EmailTool"),
    ),

    # ----------------------------------------------------------- finance
    Scenario(
        cond_query="Handle my finances for this quarter, depending on how the accounts look.",
        static_query="Set up my quarterly budget and issue the outstanding invoice.",
        probe="FinanceTool", branch=("BudgetTool", "TaxTool"),
        result=("Accounts in good standing; no unusual activity.",
                "Large capital gain realized this quarter; taxable event flagged."),
        static_gold=("BudgetTool", "InvoiceTool"),
    ),
    Scenario(
        cond_query="Close out the month on the company card, depending on what the balance says.",
        static_query="Issue this month's company card invoice and compute the tax position.",
        probe="FinanceTool", branch=("InvoiceTool", "BudgetTool"),
        result=("Card balance is well within the monthly limit.",
                "Card balance exceeds the limit; spending controls triggered."),
        static_gold=("InvoiceTool", "TaxTool"),
    ),
    Scenario(
        cond_query="Deal with the outstanding supplier invoice as the account state dictates.",
        static_query="Adjust the supplier budget line and email accounts payable the summary.",
        probe="InvoiceTool", branch=("FinanceTool", "EscalateTool"),
        result=("Invoice 20814 is 6 days overdue; routine follow-up applies.",
                "Invoice 20814 is 94 days overdue and disputed by the supplier."),
        static_gold=("BudgetTool", "EmailTool"),
    ),
    Scenario(
        cond_query="Handle this year's filing in whatever way the numbers require.",
        static_query="Pull this year's account balances and issue the final invoice.",
        probe="TaxTool", branch=("FinanceTool", "LegalTool"),
        result=("Liability computes cleanly under the standard schedule.",
                "Liability triggers an unusual-transaction threshold requiring review."),
        static_gold=("FinanceTool", "InvoiceTool"),
    ),
    Scenario(
        cond_query="Review the team's travel spend and respond to what you find.",
        static_query="Pull the team's balances and file a ticket for the travel spend review.",
        probe="BudgetTool", branch=("EmailTool", "EscalateTool"),
        result=("Travel is tracking at 71% of the annual allocation.",
                "Travel has overrun the annual allocation by 38%."),
        static_gold=("FinanceTool", "TicketTool"),
    ),
    Scenario(
        cond_query="Chase the overdue invoice in whatever manner is justified.",
        static_query="Look up the customer's billing record and message them the balance.",
        probe="InvoiceTool", branch=("EmailTool", "EscalateTool"),
        result=("The invoice is 9 days late; a reminder is the normal step.",
                "The invoice is 120 days late and the account is in collections."),
        static_gold=("BillingTool", "MessageTool"),
    ),

    # ------------------------------------------------------------- health
    Scenario(
        cond_query="I've had a headache and a cough for two days. Do what the symptoms warrant.",
        static_query="Refill my usual prescription and confirm my insurance still covers it.",
        probe="HealthTool", branch=("PharmacyTool", "DoctorTool"),
        result=("Symptoms consistent with a common cold; rest and over-the-counter care advised.",
                "Symptoms may indicate an infection requiring evaluation; see a physician."),
        static_gold=("PharmacyTool", "InsuranceTool"),
    ),
    Scenario(
        cond_query="My prescription is running low. Take the right step for the coverage situation.",
        static_query="Find a pharmacy that stocks my prescription and add the pickup to my calendar.",
        probe="InsuranceTool", branch=("PharmacyTool", "DoctorTool"),
        result=("The medication is covered at the standard copay.",
                "Coverage lapsed; a new prescription authorisation is required."),
        static_gold=("PharmacyTool", "CalendarTool"),
    ),
    Scenario(
        cond_query="I've got chest tightness after exercise. Do what's warranted.",
        static_query="Book me a physician appointment and put it on my calendar.",
        probe="HealthTool", branch=("DoctorTool", "PharmacyTool"),
        result=("Likely exertional and benign; monitor and follow up if it persists.",
                "Chest symptoms on exertion warrant prompt clinical assessment."),
        static_gold=("DoctorTool", "CalendarTool"),
    ),
    Scenario(
        cond_query="Set up my annual physical, in line with what the plan covers.",
        static_query="Book my annual physical and email me the confirmation.",
        probe="InsuranceTool", branch=("DoctorTool", "BillingTool"),
        result=("Annual preventive care is fully covered in network.",
                "The plan treats this visit as out of network with a deductible."),
        static_gold=("DoctorTool", "EmailTool"),
    ),
    Scenario(
        cond_query="Get my medication sorted according to what's in stock nearby.",
        static_query="Confirm my insurance covers the medication and email me the details.",
        probe="PharmacyTool", branch=("OrderTool", "DoctorTool"),
        result=("Two nearby pharmacies have it in stock today.",
                "The medication is unavailable locally and is on national backorder."),
        static_gold=("InsuranceTool", "EmailTool"),
    ),
    Scenario(
        cond_query="Move my appointment to suit whatever slots exist.",
        static_query="Check my calendar for next week and email the clinic my preferred times.",
        probe="DoctorTool", branch=("CalendarTool", "MessageTool"),
        result=("The clinic has openings Tuesday and Thursday afternoon.",
                "The clinic is fully booked for six weeks."),
        static_gold=("CalendarTool", "EmailTool"),
    ),

    # ------------------------------------------------- inventory / orders
    Scenario(
        cond_query="Customer wants 40 units of SKU 7731. Handle it according to what we have.",
        static_query="Place an order for 40 units of SKU 7731 and arrange shipping.",
        probe="InventoryTool", branch=("OrderTool", "SupplierTool"),
        result=("SKU 7731: 312 units in stock.", "SKU 7731: 0 units in stock; backordered."),
        static_gold=("OrderTool", "ShippingTool"),
    ),
    Scenario(
        cond_query="A reseller wants 500 units of the Q3 line. Proceed on what's available.",
        static_query="Place the reseller's 500-unit order and issue the invoice.",
        probe="InventoryTool", branch=("SupplierTool", "OrderTool"),
        result=("Q3 line: 1,840 units on hand across both warehouses.",
                "Q3 line: 60 units on hand; insufficient for the order."),
        static_gold=("OrderTool", "InvoiceTool"),
    ),
    Scenario(
        cond_query="The warehouse flagged SKU 2210. Act on the supplier's position.",
        static_query="Check stock for SKU 2210 and file a ticket for the discrepancy.",
        probe="SupplierTool", branch=("OrderTool", "EscalateTool"),
        result=("The supplier confirms a restock arriving in four days.",
                "The supplier has discontinued the line with no replacement."),
        static_gold=("InventoryTool", "TicketTool"),
    ),
    Scenario(
        cond_query="Fulfil the Nairobi order however the stock allows.",
        static_query="Place the Nairobi order and arrange the shipment.",
        probe="InventoryTool", branch=("ShippingTool", "SupplierTool"),
        result=("All line items for the Nairobi order are in stock.",
                "Two line items are out of stock; the order cannot ship complete."),
        static_gold=("OrderTool", "ShippingTool"),
    ),
    Scenario(
        cond_query="Customer changed their delivery address mid-shipment. Handle it per the parcel's status.",
        static_query="Update the customer's account address and email them the confirmation.",
        probe="ShippingTool", branch=("MessageTool", "RefundTool"),
        result=("The parcel is still at the origin depot and can be redirected.",
                "The parcel is already out for delivery to the old address."),
        static_gold=("AccountTool", "EmailTool"),
    ),
    Scenario(
        cond_query="Check on order 55812 and take the appropriate step.",
        static_query="Arrange shipping for order 55812 and email the customer the tracking number.",
        probe="OrderTool", branch=("ShippingTool", "RefundTool"),
        result=("Order 55812 is paid and ready to pick.",
                "Order 55812 failed payment authorisation twice."),
        static_gold=("ShippingTool", "EmailTool"),
    ),
    Scenario(
        cond_query="The freight is late out of Rotterdam. Act on the tracking state.",
        static_query="Email the consignee about the Rotterdam freight and re-place the short-shipped order.",
        probe="ShippingTool", branch=("EscalateTool", "MessageTool"),
        result=("The freight cleared customs and is moving, two days behind.",
                "The freight is held indefinitely in a customs inspection queue."),
        static_gold=("EmailTool", "OrderTool"),
    ),
    Scenario(
        cond_query="A parcel hasn't arrived. Do whatever the tracking situation calls for.",
        static_query="Email the customer their tracking number and file a delivery ticket.",
        probe="ShippingTool", branch=("EmailTool", "RefundTool"),
        result=("Parcel is in transit and on schedule for tomorrow.",
                "Parcel was marked lost in transit by the carrier."),
        static_gold=("EmailTool", "TicketTool"),
    ),

    # ------------------------------------------- scheduling / comms
    Scenario(
        cond_query="Set up a sync with Priya about the launch, however her schedule allows.",
        static_query="Create a meeting with Priya about the launch and email her the agenda.",
        probe="CalendarTool", branch=("EmailTool", "MessageTool"),
        result=("Priya is free Thursday 2-3pm.",
                "Priya is out of office all week; urgent items only."),
        static_gold=("MeetingTool", "EmailTool"),
    ),
    Scenario(
        cond_query="Get the design review on the books, as everyone's availability permits.",
        static_query="Create the design review meeting and file a ticket to track the outcome.",
        probe="CalendarTool", branch=("MeetingTool", "MessageTool"),
        result=("All six reviewers share a free slot Wednesday morning.",
                "No common slot exists in the next three weeks."),
        static_gold=("MeetingTool", "TicketTool"),
    ),
    Scenario(
        cond_query="Reach the on-call engineer in whatever way actually gets through.",
        static_query="Message the on-call engineer and file a ticket for the incident.",
        probe="CalendarTool", branch=("MessageTool", "EmailTool"),
        result=("The on-call engineer is active and in working hours.",
                "The on-call engineer is off-shift and unreachable until morning."),
        static_gold=("MessageTool", "TicketTool"),
    ),
    Scenario(
        cond_query="Brief the board on the incident, in the format the calendar allows.",
        static_query="Email the board the incident summary and attach the compliance documents.",
        probe="CalendarTool", branch=("MeetingTool", "EmailTool"),
        result=("The board has a standing slot tomorrow that can absorb this.",
                "The board does not convene for another five weeks."),
        static_gold=("EmailTool", "LegalTool"),
    ),
    Scenario(
        cond_query="Set up the retro in the form that works given the invite status.",
        static_query="Check the team calendar for retro slots and email the invitations.",
        probe="MeetingTool", branch=("MessageTool", "EmailTool"),
        result=("The existing retro invite is live with eight acceptances.",
                "The retro invite was never sent; no one has been notified."),
        static_gold=("CalendarTool", "EmailTool"),
    ),

    # -------------------------------------------------------------- devops
    Scenario(
        cond_query="Ship the current build, or don't, depending on where it stands.",
        static_query="Deploy the current build and file a release ticket.",
        probe="RepoTool", branch=("DeployTool", "RollbackTool"),
        result=("Build 8812: all 1,204 tests passing.",
                "Build 8812: 37 tests failing in the payments module."),
        static_gold=("DeployTool", "TicketTool"),
    ),
    Scenario(
        cond_query="Push 9.2 to production or hold it, according to the pipeline.",
        static_query="Deploy release 9.2 to production and message the team when it lands.",
        probe="RepoTool", branch=("DeployTool", "TicketTool"),
        result=("Release 9.2: pipeline green across all stages.",
                "Release 9.2: the integration stage is red on three suites."),
        static_gold=("DeployTool", "MessageTool"),
    ),
    Scenario(
        cond_query="Production latency spiked after the last release. Respond to what the repo shows.",
        static_query="Roll production back to the previous build and file an incident ticket.",
        probe="RepoTool", branch=("TicketTool", "RollbackTool"),
        result=("The last release touched only static assets; latency is unrelated.",
                "The last release shipped an unindexed query path on the hot route."),
        static_gold=("RollbackTool", "TicketTool"),
    ),
    Scenario(
        cond_query="Decide the fate of the hotfix branch based on its test state.",
        static_query="File a ticket for the hotfix branch and message the reviewer.",
        probe="RepoTool", branch=("DeployTool", "EscalateTool"),
        result=("The hotfix branch is green and behind a feature flag.",
                "The hotfix branch fails the migration test and has no rollback path."),
        static_gold=("TicketTool", "MessageTool"),
    ),
    Scenario(
        cond_query="Finish the rollout in the manner the deploy state allows.",
        static_query="Check the repository's build status and file a ticket for the rollout.",
        probe="DeployTool", branch=("RollbackTool", "TicketTool"),
        result=("The rollout reached 100% of hosts with no error-rate change.",
                "The rollout stalled at 40% with elevated 5xx on the canary."),
        static_gold=("RepoTool", "TicketTool"),
    ),

    # ----------------------------------------------- billing / support
    Scenario(
        cond_query="A customer is disputing last month's charge. Act on what the record shows.",
        static_query="Issue the approved refund for last month's charge and email the customer.",
        probe="BillingTool", branch=("RefundTool", "EscalateTool"),
        result=("Charge confirmed as a duplicate; refund is within policy.",
                "Charge is valid and already disputed twice; policy requires supervisor review."),
        static_gold=("RefundTool", "EmailTool"),
    ),
    Scenario(
        cond_query="A customer says they were double-charged. Do what the ledger supports.",
        static_query="Refund the duplicate charge and file a ticket for the billing defect.",
        probe="BillingTool", branch=("RefundTool", "EmailTool"),
        result=("Two identical charges posted the same minute; one is in error.",
                "Only one charge posted; the second is an authorisation hold."),
        static_gold=("RefundTool", "TicketTool"),
    ),
    Scenario(
        cond_query="This enterprise account is threatening to churn. Act on their billing history.",
        static_query="Email the enterprise account owner and set up a retention meeting.",
        probe="BillingTool", branch=("EscalateTool", "RefundTool"),
        result=("The account has paid on time for three years at growing volume.",
                "The account has three unresolved billing complaints this quarter."),
        static_gold=("EmailTool", "MeetingTool"),
    ),
    Scenario(
        cond_query="Resolve the disputed subscription charge however the record justifies.",
        static_query="Update the customer's account plan and email them the revised terms.",
        probe="BillingTool", branch=("RefundTool", "LegalTool"),
        result=("The subscription was cancelled before renewal; the charge is refundable.",
                "The contract has a non-cancellable annual term with signed acceptance."),
        static_gold=("AccountTool", "EmailTool"),
    ),
    Scenario(
        cond_query="Follow up on ticket 4417 as its status requires.",
        static_query="Email the requester about ticket 4417 and check their account settings.",
        probe="TicketTool", branch=("EscalateTool", "MessageTool"),
        result=("Ticket 4417 is resolved and awaiting customer confirmation.",
                "Ticket 4417 has breached its SLA by eleven days."),
        static_gold=("EmailTool", "AccountTool"),
    ),
    Scenario(
        cond_query="A user wants their plan changed. Do what their account permits.",
        static_query="Look up the user's billing record and email them the plan comparison.",
        probe="AccountTool", branch=("BillingTool", "EscalateTool"),
        result=("The account is self-serve and eligible for an immediate plan change.",
                "The account is under a managed contract; changes need approval."),
        static_gold=("BillingTool", "EmailTool"),
    ),

    # ----------------------------------------------- security / account
    Scenario(
        cond_query="A user can't sign in. Take the appropriate action for their account state.",
        static_query="Reset this user's password and email them the confirmation.",
        probe="SecurityTool", branch=("ResetTool", "EscalateTool"),
        result=("No suspicious activity; a routine credential reset is appropriate.",
                "Account shows sign-ins from three countries in one hour; possible compromise."),
        static_gold=("ResetTool", "EmailTool"),
    ),
    Scenario(
        cond_query="Someone reported a suspicious login email. Take the step the account state calls for.",
        # The static query must NOT ask for the probe tool's own job: an earlier
        # draft said "check this account's recent sign-ins", which is
        # SecurityTool, i.e. the probe -- making probing non-wasteful at alpha=0
        # and the over-gathering metric meaningless. validate_bank caught it.
        static_query="Look up this user's account settings and message them a summary.",
        probe="SecurityTool", branch=("AccountTool", "ResetTool"),
        result=("The sign-in was the user's own device on a new network.",
                "The sign-in came from a host on a known credential-stuffing range."),
        static_gold=("AccountTool", "MessageTool"),
    ),
    Scenario(
        cond_query="A departing employee still has access. Handle it as the audit indicates.",
        static_query="Update the departing employee's account settings and pull their HR record.",
        probe="SecurityTool", branch=("AccountTool", "EscalateTool"),
        result=("Access is limited to routine tools already scheduled for removal.",
                "The account retains production credentials used after the departure date."),
        static_gold=("AccountTool", "HRTool"),
    ),
    Scenario(
        cond_query="Lock down this account to the degree the sign-in log warrants.",
        static_query="Reset the account credential and message the owner that it changed.",
        probe="SecurityTool", branch=("ResetTool", "AccountTool"),
        result=("Sign-ins are all from the registered device and region.",
                "Sign-ins show six failed attempts then a success from a new country."),
        static_gold=("ResetTool", "MessageTool"),
    ),

    # ------------------------------------------- HR / legal / payroll
    Scenario(
        cond_query="An employee asked about taking leave next month. Respond based on their balance.",
        static_query="Enrol this employee in the compliance training and email them the schedule.",
        probe="HRTool", branch=("CalendarTool", "PayrollTool"),
        result=("Employee has 14 days of leave remaining; no payroll impact.",
                "Employee has 0 days remaining; further leave would be unpaid."),
        static_gold=("TrainingTool", "EmailTool"),
    ),
    Scenario(
        cond_query="Review the vendor contract renewal and proceed as the terms require.",
        static_query="Update the vendor's account record and file a ticket for the renewal review.",
        probe="LegalTool", branch=("InvoiceTool", "EscalateTool"),
        result=("Contract auto-renews at the same rate; standard invoicing applies.",
                "Contract contains a non-standard indemnity clause requiring sign-off."),
        static_gold=("AccountTool", "TicketTool"),
    ),
    Scenario(
        cond_query="An employee's overtime looks wrong. Fix it per their record.",
        static_query="Correct this employee's payroll entry and email them the revised slip.",
        probe="HRTool", branch=("PayrollTool", "EscalateTool"),
        result=("The record shows 6 approved overtime hours matching the claim.",
                "The record shows no approved overtime for the period claimed."),
        static_gold=("PayrollTool", "EmailTool"),
    ),
    Scenario(
        cond_query="Onboard the new analyst in the way their start date allows.",
        static_query="Enrol the new analyst in onboarding training and create their account.",
        probe="HRTool", branch=("TrainingTool", "CalendarTool"),
        result=("The start date is confirmed for the 1st; standard onboarding applies.",
                "The start date slipped by a month pending a background check."),
        static_gold=("TrainingTool", "AccountTool"),
    ),
    Scenario(
        cond_query="The NDA came back marked up. Proceed as the clauses require.",
        static_query="File a ticket for the NDA markup and email counsel the summary.",
        probe="LegalTool", branch=("EscalateTool", "EmailTool"),
        result=("The markup is limited to the notice period; within delegated authority.",
                "The markup strikes the confidentiality term entirely."),
        static_gold=("TicketTool", "EmailTool"),
    ),
    Scenario(
        cond_query="Handle the contractor's extension according to the contract terms.",
        static_query="Pull the contractor's HR record and set up their payroll entry.",
        probe="LegalTool", branch=("HRTool", "EscalateTool"),
        result=("The agreement permits a six-month extension on the same terms.",
                "The agreement caps total engagement at twelve months, already reached."),
        static_gold=("HRTool", "PayrollTool"),
    ),
    Scenario(
        cond_query="Run this month's payroll, adjusting for whatever the records show.",
        static_query="Pull the employee records for this month and email the payroll summary.",
        probe="PayrollTool", branch=("HRTool", "EscalateTool"),
        result=("All entries reconcile against the approved headcount.",
                "Two entries pay employees with no active HR record."),
        static_gold=("HRTool", "EmailTool"),
    ),
    Scenario(
        cond_query="Assign the mandatory security course as the employee's history dictates.",
        static_query="Pull the employee's record and add the course dates to their calendar.",
        probe="TrainingTool", branch=("EmailTool", "HRTool"),
        result=("The employee completed the prerequisite course last year.",
                "The employee has never completed the prerequisite and is overdue."),
        static_gold=("HRTool", "CalendarTool"),
    ),

    # ------------------------------------------------------ news-driven
    Scenario(
        cond_query="Brief me on the supplier strike coverage and act accordingly.",
        static_query="Check stock levels for the affected lines and email the procurement team.",
        probe="NewsTool", branch=("SupplierTool", "EscalateTool"),
        result=("Coverage reports the strike was settled before any stoppage.",
                "Coverage reports an indefinite stoppage at both main ports."),
        static_gold=("InventoryTool", "EmailTool"),
    ),
    Scenario(
        cond_query="There's chatter about our product recall. Respond as the coverage warrants.",
        static_query="File a ticket about the recall chatter and email the comms lead.",
        probe="NewsTool", branch=("EscalateTool", "MessageTool"),
        result=("Only one trade blog mentions it, with no pickup elsewhere.",
                "Three national outlets are running it as a lead consumer story."),
        static_gold=("TicketTool", "EmailTool"),
    ),
]


def validate_bank(scenarios: List[Scenario] = None) -> None:
    """Fails loudly on the design violations that previously went unnoticed.

    Runs at import, so an invalid bank cannot reach an experiment. Every
    assertion here corresponds to a bug that was found in review and silently
    corrupted a reported metric.
    """
    scenarios = SCENARIOS if scenarios is None else scenarios
    assert len(scenarios) >= 2, "bank is empty"
    seen_cond, seen_static = set(), set()
    for i, s in enumerate(scenarios):
        assert s.probe not in s.static_gold, (
            f"scenario {i}: probe {s.probe!r} is in static_gold -- probing would "
            f"not be wasteful at alpha=0 and over-gathering becomes unmeasurable")
        assert s.branch[0] != s.branch[1], f"scenario {i}: branches must differ"
        assert s.probe not in s.branch, f"scenario {i}: probe must not be a branch"
        assert len(set(s.static_gold)) == 2, f"scenario {i}: static_gold must be 2 distinct tools"
        assert s.cond_query != s.static_query, f"scenario {i}: queries must differ"
        assert s.result[0] != s.result[1], f"scenario {i}: probe results must differ"
        for t in (s.probe,) + tuple(s.branch) + tuple(s.static_gold):
            assert t in TOOLS, f"scenario {i}: unknown tool {t!r}"
            assert t != "DONE", f"scenario {i}: DONE cannot be a gold tool"
        # Duplicate query text would make two scenarios indistinguishable from
        # the observation, so no policy could be correct on both.
        assert s.cond_query not in seen_cond, f"scenario {i}: duplicate cond_query"
        assert s.static_query not in seen_static, f"scenario {i}: duplicate static_query"
        seen_cond.add(s.cond_query)
        seen_static.add(s.static_query)


def split(n_test: int = 20, seed: int = 0):
    """Deterministic scenario-level train/test split, returned as index lists.

    A scenario-level split is a GENERALIZATION test and a hard one: the gold
    tools of a held-out scenario never appear with its query during training,
    so a policy can only succeed by inferring the probe and branch from the
    text. An LLM can attempt that zero-shot; a PPO agent over frozen
    embeddings largely cannot. That asymmetry is informative but it changes
    the claim, so nothing here imposes the split -- callers opt in.
    """
    import numpy as _np
    idx = _np.arange(len(SCENARIOS))
    _np.random.default_rng(seed).shuffle(idx)
    test = sorted(int(i) for i in idx[:n_test])
    train = sorted(int(i) for i in idx[n_test:])
    return train, test


validate_bank()
