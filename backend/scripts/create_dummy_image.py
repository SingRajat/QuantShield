from PIL import Image, ImageDraw, ImageFont

def create_dummy_etf_image(output_path="dummy_etf.png"):
    # Create a white image
    img = Image.new('RGB', (800, 600), color='white')
    d = ImageDraw.Draw(img)
    
    # Try to load a font, otherwise use default
    try:
        font = ImageFont.truetype("arial.ttf", 20)
        title_font = ImageFont.truetype("arial.ttf", 36)
    except IOError:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()
        
    # Draw title
    d.text((50, 50), "XYZ Technology ETF Factsheet", fill="black", font=title_font)
    
    # Draw some introductory text
    intro = "This ETF seeks to track the investment results of an index composed of US equities in the technology sector."
    d.text((50, 100), intro, fill="black", font=font)
    
    # Draw Top Holdings table header
    d.text((50, 200), "Top 10 Holdings", fill="black", font=title_font)
    d.text((50, 250), "Ticker      Company Name                           Weight (%)", fill="black", font=font)
    d.line([(50, 280), (750, 280)], fill="black", width=2)
    
    # Draw holdings rows
    holdings = [
        ("AAPL", "Apple Inc.", "22.50%"),
        ("MSFT", "Microsoft Corp.", "18.30%"),
        ("NVDA", "NVIDIA Corporation", "5.40%"),
        ("AVGO", "Broadcom Inc.", "4.20%"),
        ("CSCO", "Cisco Systems, Inc.", "3.10%")
    ]
    
    y = 300
    for ticker, name, weight in holdings:
        d.text((50, y), ticker, fill="black", font=font)
        d.text((150, y), name, fill="black", font=font)
        d.text((550, y), weight, fill="black", font=font)
        y += 35
        
    img.save(output_path)
    print(f"Saved dummy image to {output_path}")

if __name__ == "__main__":
    create_dummy_etf_image()
