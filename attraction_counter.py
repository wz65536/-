import os
import json

def read_province_data(file_path):
    """Read attraction data from a province file"""
    attractions = []
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        lines = lines[2:]  # Skip header and separator
        
        for line in lines:
            parts = line.strip().split(";")
            if len(parts) < 7:
                continue
                
            attraction = {
                "name": parts[0].strip(),
                "location": parts[1].strip(),
                "rating": parts[2].strip(),
                "heat": parts[3].strip(),
                "reputation": parts[4].strip(),
                "tags": parts[5].strip(),
                "reviews": parts[6].strip()
            }
            attractions.append(attraction)
    
    return attractions

def count_high_rated_attractions():
    """Count 4A and above attractions for each province"""
    data_dir = "data"
    province_counts = {}
    
    # Skip the combined file
    province_files = [f for f in os.listdir(data_dir) if f.endswith(".txt") and f != "全国景点.txt"]
    
    print(f"Processing {len(province_files)} province files...")
    
    for file_name in province_files:
        province_name = file_name.replace(".txt", "")
        file_path = os.path.join(data_dir, file_name)
        
        attractions = read_province_data(file_path)
        
        # Count 4A and 5A attractions
        high_rated_count = sum(1 for attr in attractions if attr["rating"] in ["4A", "5A"])
        
        # Store the count
        province_counts[province_name] = high_rated_count
        print(f"{province_name}: {high_rated_count} 4A+ attractions")
    
    # Map city names to province names for consistency with the map
    city_to_province = {
        "北京": "北京",
        "天津": "天津",
        "上海": "上海",
        "重庆": "重庆",
        "香港": "香港",
        "澳门": "澳门",
        "台湾": "台湾"
    }
    
    # Adjust counts for cities that are treated as provinces
    for city, province in city_to_province.items():
        if city in province_counts:
            if province != city and province in province_counts:
                # If both city and province exist, add city count to province
                province_counts[province] += province_counts[city]
                del province_counts[city]
                print(f"Added {city} count to {province}")
            elif province != city:
                # If only city exists, rename to province
                province_counts[province] = province_counts[city]
                del province_counts[city]
                print(f"Renamed {city} to {province}")
    
    # Save the counts to a JavaScript file
    with open("attraction_counts.js", "w", encoding="utf-8") as f:
        f.write("var attractionCounts = " + json.dumps(province_counts, ensure_ascii=False, indent=2) + ";")
    
    print(f"\nCounted 4A+ attractions for {len(province_counts)} provinces")
    return province_counts

if __name__ == "__main__":
    counts = count_high_rated_attractions()
    print("\nProvince counts (sorted by count):")
    for province, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{province}: {count}") 