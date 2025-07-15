cancel_order
- **Definition:** 
- **Positive Examples:**  
  - "would it be possible to cancel the order i made? "  
  - " problem with canceling the order i made" 
- **Negative Examples (edge cases) :**  
  - "what is the fee if i cancel?" → NOT cancel_order (that’s check_cancellation_fee)



change_order
- **Definition:** Requests to modify/alter an existing placed order,adding/removing items.
- **Positive Examples:**  
  - "i need help adding several items"  
  - "problems with adding an item to an order" 
- **Negative Examples (edge cases) :**  
  - "where is my order?" → NOT change_order (that’s track_order)

  

change_shipping_address
- **Definition:** Requests to modify the delivery address for an existing order or an address stored in the user's account. 
- **Positive Examples:**  
  - "could you tell me more about changing my shipping address?"  
  - "what do i need to do to correct the address?" 
- **Negative Examples (edge cases) :**  
  - "how do I add a new shipping address to my profile?" → NOT change_shipping_address (that’s set_up_shipping_address)


check_cancellation_fee
- **Definition:** 
- **Positive Examples:**  
  - "i want assistance to check the withdrawal fee"  
  - "i cannot find the termination charge, can i get some help ?" 
- **Negative Examples (edge cases) :**  
  - "I need to cancel my order." → NOT check_cancellation_fee (that’s cancel_order)


check_invoice
- **Definition:** Inquiries about any charges, fees, or penalties incurred when canceling an order or service. 
- **Positive Examples:**  
  - "help me taking a quick look at the invoices from last month"  
  - "can you help me take a quick look at my invoice?" 
- **Negative Examples (edge cases) :**  
  - "Can I get a breakdown of the charges on my last invoice?" → NOT get_invoice (that’s check_invoice)



check_payment_methods
- **Definition:** Queries about the types of payment options accepted by the service or available for use.
- **Positive Examples:**  
  - "i dont know how i can see the payment options"  
  - "i would like to check what payment methods are available" 
- **Negative Examples (edge cases) :**  
  - "my payment isn't going through." → NOT check_payment_methods (that’s payment_issue)



check_refund_policy
- **Definition:**  Inquiries about the rules, terms, or conditions for obtaining a refund.
- **Positive Examples:**  
  - "help me check how long refunds usually take"  
  - "i need help checking in what cases can i ask to be refunded" 
- **Negative Examples (edge cases) :**  
  - "I want to get a refund for my order." → NOT check_refund_policy (that’s get_refund)



complaint
- **Definition:** Expressing strong dissatisfaction, reporting a significant grievance, or formally escalating an issue. 
- **Positive Examples:**  
  - "i try to file a complaint against your company"  
  - "where could i file a reclamation" 
- **Negative Examples (edge cases) :**  
  - "My order hasn't arrived yet." → NOT complaint (that’s track_order)

   While this expresses dissatisfaction, the primary intent is to get information about the order's status, not to formally lodge a grievance.



contact_customer_service
- **Definition:**Requests for general contact information or methods (e.g., email, phone number) to reach customer support. 
- **Positive Examples:**  
  - "what is your customer service mail?"  
  - "i do not knwo how to get in touch with customer support" 
- **Negative Examples (edge cases) :**  
  - "I need to talk to a person right now." → NOT contact_customer_service (that’s contact_human_agent)



contact_human_agent
- **Definition:**  Explicit requests to speak directly with a live person, customer representative, or human agent.
- **Positive Examples:**  
  - "help me to speak with someone "  
  - "can i chat with a human agent? " 
- **Negative Examples (edge cases) :**  
  - "What is your customer service email address?" → NOT contact_human_agent (that’s contact_customer_service)



create_account
- **Definition:**Requests or inquiries about registering for a new user account or initiating the sign-up process. 
- **Positive Examples:**  
  - "how do i create an account?"  
  - "i want help to open an account" 
- **Negative Examples (edge cases) :**  
  - "I forgot my password, can you help me log in?" → NOT create_account (that’s recover_password)




delete_account
- **Definition:**Requests or inquiries about permanently closing, removing, or deactivating a user account. 
- **Positive Examples:**  
  - "i would like to know about a user account deletion"  
  - "i want help deleting the account" 
- **Negative Examples (edge cases) :**  
  - "I need to change my email address on my account." → NOT delete_account (that’s edit_account)



delivery_options
- **Definition:** Inquiries about available shipping methods, carriers, or service levels (e.g., standard, express, in-store pick-up).
- **Positive Examples:**  
  - "i am calling to check what options for delivery i have"  
  - "where to check the options for delivery?" 
- **Negative Examples (edge cases) :**  
  - "When will my package arrive?" → NOT delivery_options (that’s delivery_period)



delivery_period
- **Definition:** Inquiries about the estimated time of arrival, shipping duration, or general expected delivery window for orders or services.
- **Positive Examples:**  
  - "could you help me check when my delivery is going to arrive?"  
  - "i need assistance checking how soon can i expect the item" 
- **Negative Examples (edge cases) :**  
  - "Where is my package right now?" → NOT delivery_period (that’s track_order)



edit_account
- **Definition:**Requests to change, update, or modify personal details, preferences, or settings within an existing user account. 
- **Positive Examples:**  
  - "is it possible to edit my personal information?"  
  - "could you help me edit my account?" 
- **Negative Examples (edge cases) :**  
  - "I want to delete my account completely." → NOT edit_account (that’s delete_account)



get_invoice
- **Definition:** Requests to retrieve, download, or be sent a specific bill, receipt, or invoice for a past transaction.
- **Positive Examples:**  
  - "where do i get the invoice from 8 months ago?"  
  - "i would like to get the invoice from one purchases ago" 
- **Negative Examples (edge cases) :**  
  - "Is my last invoice correct?" → NOT get_invoice (that’s check_invoice)



get_refund
- **Definition:** Requests or actions to initiate, process, or formally receive a monetary refund for a specific purchase, service, or issue
- **Positive Examples:**  
  - "how could i get my money back?"  
  - "where to get a refund?" 
- **Negative Examples (edge cases) :**  
  - Has my refund been processed yet?" → NOT get_refund (that’s track_refund)



newsletter_subscription
- **Definition:** Requests or inquiries related to subscribing to, unsubscribing from, or managing settings for email newsletters, promotional lists, or free email communications.
- **Positive Examples:**  
  - "help me to unsubscribe to your corporate newsletter"  
  - "i want to cancel the subscription to your company newsletter" 
- **Negative Examples (edge cases) :**  
  - "I have a problem with my monthly service subscription payment." → NOT newsletter_subscription (that’s payment_issue)



payment_issue
- **Definition:**Reports of problems, errors, or discrepancies related to payments, charges, or financial transactions (e.g., failed payments, double billing, incorrect charges, charges for cancelled items, missing refunds)
- **Positive Examples:**  
  - "i want to report issues with payments"  
  - "help to solve issues with payment" 
- **Negative Examples (edge cases) :**  
  - "What forms of payment do you accept?" → NOT payment_issue (that’s check_payment_methods)



place_order
- **Definition:** Requests or inquiries about initiating a new purchase, adding items to a shopping cart, or completing a transaction to buy goods or services. 
- **Positive Examples:**  
  - "could you help me buy some items?"  
  - "where to make an order?" 
- **Negative Examples (edge cases) :**  
  - "When will my order arrive?" → NOT place_order (that’s track_order)


recover_password
- **Definition:**Requests or inquiries about regaining access to an account due to a forgotten, lost, or inaccessible password/login credentials. 
- **Positive Examples:**  
  - "how could i find information about forgotten passwords?"  
  - " " 
- **Negative Examples (edge cases) :**  
  - "I want to change my password." → NOT recover_password (that’s edit_account)



registration_problems
- **Definition:** Reports of errors, difficulties, or technical issues encountered specifically during the account creation or sign-up process. 
- **Positive Examples:**  
  - "issue with signup"  
  - "i am trying to report registration issues" 
- **Negative Examples (edge cases) :**  
  - "I forgot my password and can't log in." → NOT registration_problems (that’s recover_password)



review
- **Definition:** Requests or statements related to providing feedback, testimonials, ratings, or opinions on products, services, or the overall company experience.
- **Positive Examples:**  
  - "help me submitting some feedback"  
  - "help leaving a review for your company" 
- **Negative Examples (edge cases) :**  
  - "I need to file an official complaint about a billing error." → NOT review (that’s complaint)



set_up_shipping_address
- **Definition:** Requests or inquiries about adding a new shipping address to an account or profile, or establishing a primary delivery address for the first time.
- **Positive Examples:**  
  - "can you help me to set up another shipping address?"  
  - "i have an issue setting another shipping address up" 
- **Negative Examples (edge cases) :**  
  - "I need to update my shipping address for my current order." → NOT set_up_shipping_address (that’s change_shipping_address)



switch_account
- **Definition:**Requests or inquiries about logging out of one user account and logging into a different existing user account, or managing access to multiple distinct profiles. 
- **Positive Examples:**  
  - "i need help changing to a new profile"  
  - "where to switch to a new user account" 
- **Negative Examples (edge cases) :**  
  - "I want to change my current account's password." → NOT switch_account (that’s edit_account or recover_password)



track_order
- **Definition:**Inquiries about the current status, location, or estimated delivery time of a specific, already placed order. 
- **Positive Examples:**  
  - "where to check the status of my order?  
  - "i would like to check my order eta" 
- **Negative Examples (edge cases) :**  
  - "Can I add another item to my recent order?" → NOT track_order (that’s change_order)



track_refund 
- **Definition:** Inquiries about the current status or progress of a refund that has already been initiated or is expected 
- **Positive Examples:**  
  - "i need assistance checking the status of the refund"  
  - "what do i need to do to track the refund?" 
- **Negative Examples (edge cases) :**  
  - "I want to get a refund for my last purchase." → NOT track_refund (that’s get_refund)