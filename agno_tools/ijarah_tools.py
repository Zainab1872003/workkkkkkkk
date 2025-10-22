# agno_tools/ijarah_tools.py
"""
Tools for Ijarah Agent - Islamic financing calculations
"""
import logging

logger = logging.getLogger(__name__)


def calculate_bike_ijarah(
    bike_price: float,
    down_payment: float,
    tenure_months: int,
    profit_rate: float = 15.0
) -> str:
    """
    Calculate Islamic bike financing monthly installments.
    
    Args:
        bike_price: Total bike price in PKR
        down_payment: Down payment amount in PKR
        tenure_months: Financing tenure in months
        profit_rate: Annual profit rate percentage
    
    Returns:
        Detailed calculation breakdown
    """
    try:
        logger.info(f"💰 calculate_bike_ijarah")
        
        bike_price = float(bike_price)
        down_payment = float(down_payment)
        tenure_months = int(tenure_months)
        profit_rate = float(profit_rate)
        
        if bike_price <= 0 or down_payment < 0 or down_payment >= bike_price:
            return "❌ Invalid inputs"
        if tenure_months <= 0 or tenure_months > 72:
            return "❌ Tenure must be 1-72 months"
        
        financed = bike_price - down_payment
        profit = (financed * profit_rate * tenure_months) / (100 * 12)
        monthly = (financed + profit) / tenure_months
        
        return f"""**🏍️ Bike Ijarah Calculation**
• Price: PKR {bike_price:,.0f}
• Down Payment: PKR {down_payment:,.0f}
• Financed Amount: PKR {financed:,.0f}
• Profit Rate: {profit_rate:.1f}% per annum
• Tenure: {tenure_months} months
• **Monthly Installment: PKR {monthly:,.2f}**"""
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return f"Error: {str(e)}"


def compare_financing_options(
    bike_price: float,
    down_payment: float,
    tenure_options: str = "12,24,36"
) -> str:
    """
    Compare monthly payments for different tenure options.
    
    Args:
        bike_price: Total bike price
        down_payment: Down payment amount
        tenure_options: Comma-separated tenure options (e.g., "12,24,36")
    
    Returns:
        Comparison table
    """
    try:
        logger.info(f"📊 compare_financing_options")
        
        tenures = [int(t.strip()) for t in tenure_options.split(",")]
        
        result = f"**Financing Comparison for PKR {bike_price:,.0f}**\n\n"
        for tenure in tenures:
            financed = bike_price - down_payment
            profit = (financed * 15.0 * tenure) / (100 * 12)
            monthly = (financed + profit) / tenure
            result += f"• **{tenure} months:** PKR {monthly:,.2f}/month\n"
        
        return result
    except Exception as e:
        return f"Error: {str(e)}"


def calculate_early_settlement(
    original_amount: float,
    monthly_payment: float,
    months_paid: int,
    total_months: int
) -> str:
    """
    Calculate early settlement amount for Ijarah contract.
    
    Args:
        original_amount: Original financed amount
        monthly_payment: Monthly installment
        months_paid: Number of months already paid
        total_months: Total tenure in months
    
    Returns:
        Settlement calculation
    """
    try:
        logger.info(f"💵 calculate_early_settlement")
        
        remaining_months = total_months - months_paid
        remaining_principal = original_amount * (remaining_months / total_months)
        
        return f"""**Early Settlement Calculation**
• Months Paid: {months_paid}/{total_months}
• Remaining Principal: PKR {remaining_principal:,.2f}
• Settlement Amount: PKR {remaining_principal:,.2f}
(Principal only - profit waived on early settlement)"""
        
    except Exception as e:
        return f"Error: {str(e)}"


# Tool registry for Ijarah Agent
IJARAH_TOOLS = {
    "calculate_bike_ijarah": calculate_bike_ijarah,
    "compare_financing_options": compare_financing_options,
    "calculate_early_settlement": calculate_early_settlement,
}
