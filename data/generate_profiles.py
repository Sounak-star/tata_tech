import json
import random

# Existing hardcoded profiles that must be preserved
existing = {
  "ravi": {
    "id": "ravi",
    "name": "Ravi Kumar",
    "role": "Senior Excavator Operator",
    "hearing": "impaired",
    "color_vision": "normal",
    "language": "hi",
    "experience": "expert",
    "photo": "ravi.jpg",
    "calibration": {
      "ear_baseline": 0.29,
      "perclos_baseline": 0.06,
      "blink_rate_baseline": 14.0,
      "calibrated": True
    },
    "notes": "Hard of hearing — alerts must not rely on sound (buzz + flash)."
  },
  "priya": {
    "id": "priya",
    "name": "Priya Sharma",
    "role": "Trainee Operator",
    "hearing": "normal",
    "color_vision": "normal",
    "language": "hi",
    "experience": "trainee",
    "photo": "priya.jpg",
    "calibration": {
      "ear_baseline": 0.31,
      "perclos_baseline": 0.05,
      "blink_rate_baseline": 16.0,
      "calibrated": True
    },
    "notes": "Trainee — earlier warnings, simpler UI, fewer gauges."
  },
  "arjun": {
    "id": "arjun",
    "name": "Arjun Mehta",
    "role": "Crane Operator",
    "hearing": "normal",
    "color_vision": "deuteranopia",
    "language": "ta",
    "experience": "intermediate",
    "photo": "arjun.jpg",
    "calibration": {
      "ear_baseline": 0.28,
      "perclos_baseline": 0.07,
      "blink_rate_baseline": 13.0,
      "calibrated": True
    },
    "notes": "Colour-blind (deuteranopia) — use blue/white + icons, never red-vs-green alone."
  }
}

first_names = [
    "Aarav", "Aditya", "Amit", "Anil", "Arvind", "Deepak", "Dev", "Hari", "Jaidev", "Karan",
    "Madhav", "Manoj", "Nikhil", "Pranav", "Rajesh", "Rohan", "Sanjay", "Siddharth", "Vijay", "Vikram",
    "Ananya", "Divya", "Gauri", "Jyoti", "Kavita", "Meera", "Neha", "Pooja", "Ritu", "Sunita",
    "Suresh", "Ramesh", "Naresh", "Kailash", "Kamlesh", "Harish", "Dinesh", "Ganesh", "Mahesh", "Umesh",
    "Rahul", "Rohit", "Saurabh", "Abhishek", "Manish", "Alok", "Sandeep", "Vikas", "Ashok", "Srikant",
    "Balaji", "Karthik", "Rangan", "Srinivas", "Murugan", "Raghavan", "Venkatesh", "Ramakrishnan", "Sanjay"
]

last_names = [
    "Patel", "Sharma", "Verma", "Gupta", "Joshi", "Mehta", "Rao", "Nair", "Reddy", "Choudhury",
    "Singh", "Kumar", "Mishra", "Pandey", "Yadav", "Dubey", "Trivedi", "Deshmukh", "Kulkarni", "Joshi",
    "Iyer", "Iyengar", "Pillai", "Naicker", "Shetty", "Gowda", "Hegde", "Menon", "Jha", "Prasad",
    "Subramanian", "Murthy", "Krishnan", "Nambiar", "Balakrishnan", "Suryanarayanan"
]

roles = [
    "Excavator Operator", "Crane Operator", "Bulldozer Operator", "Dumper Driver",
    "Loader Operator", "Safety Inspector", "Backhoe Operator", "Grader Operator"
]

languages = ["en", "hi", "ta", "te"]
experiences = ["trainee", "intermediate", "expert"]
color_visions = ["normal", "deuteranopia", "protanopia", "tritanopia"]

profiles = {}
profiles.update(existing)

random.seed(42)
for i in range(1, 118):
    op_id = f"op_{i:03d}"
    first = random.choice(first_names)
    last = random.choice(last_names)
    name = f"{first} {last}"
    
    # Avoid duplicate name with existing
    if name in ["Ravi Kumar", "Priya Sharma", "Arjun Mehta"]:
        name = f"{first} Kumar {last}"
        
    role = random.choice(roles)
    lang = random.choice(languages)
    exp = random.choice(experiences)
    
    # 15% chance of hearing impairment, 15% chance of color blindness
    hearing = "impaired" if random.random() < 0.15 else "normal"
    color_vis = random.choice(color_visions[1:]) if random.random() < 0.15 else "normal"
    
    notes_list = []
    if hearing == "impaired":
        notes_list.append("Hard of hearing — haptic + flash alerts.")
    if color_vis != "normal":
        notes_list.append(f"Colour-blind ({color_vis}) — high contrast + icons.")
    if exp == "trainee":
        notes_list.append("Trainee — conservative thresholds, simpler UI.")
    notes = " ".join(notes_list) if notes_list else "Standard operating profile."
    
    profiles[op_id] = {
        "id": op_id,
        "name": name,
        "role": role,
        "hearing": hearing,
        "color_vision": color_vis,
        "language": lang,
        "experience": exp,
        "photo": f"{op_id}.jpg",
        "calibration": {
            "ear_baseline": round(random.uniform(0.26, 0.33), 3),
            "perclos_baseline": round(random.uniform(0.04, 0.08), 3),
            "blink_rate_baseline": round(random.uniform(11.0, 18.0), 1),
            "calibrated": True
        },
        "notes": notes
    }

with open("data/profiles.json", "w", encoding="utf-8") as f:
    json.dump(profiles, f, indent=2)

print(f"Successfully generated {len(profiles)} profiles in data/profiles.json")
