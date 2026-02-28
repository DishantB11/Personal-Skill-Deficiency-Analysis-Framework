import pandas as pd
import random

ROLES = [
    "Backend Developer",
    "Frontend Developer",
    "Data Analyst",
    "Data Scientist",
    "Cloud Engineer",
    "DevOps Engineer",
    "Mobile Developer",
    "Cybersecurity Analyst"
]

def r(low, high):
    return random.randint(low, high)

def generate_samples(role, n=100):
    data = []

    for _ in range(n):
        sample = {}

        # Base moderate values
        sample["Python"] = r(2, 4)
        sample["Java"] = r(2, 4)
        sample["JavaScript"] = r(2, 4)
        sample["SQL"] = r(2, 4)
        sample["DSA"] = r(2, 4)
        sample["Cloud"] = r(2, 4)
        sample["DevOps"] = r(2, 4)
        sample["Networking"] = r(2, 4)
        sample["DataAnalysis"] = r(2, 4)
        sample["Communication"] = r(2, 5)

        # Academic
        sample["CGPA"] = round(random.uniform(6.5, 9.5), 2)
        sample["Projects"] = r(1, 5)

        # Performance
        sample["EasySolved"] = r(20, 120)
        sample["MediumSolved"] = r(10, 80)
        sample["HardSolved"] = r(5, 30)

        # Controlled role bias (stronger than before)
        if role == "Backend Developer":
            sample["Python"] = r(4, 5)
            sample["DSA"] = r(4, 5)
            sample["MediumSolved"] = r(40, 100)

        elif role == "Frontend Developer":
            sample["JavaScript"] = r(4, 5)
            sample["Projects"] = r(3, 6)

        elif role == "Data Analyst":
            sample["SQL"] = r(4, 5)
            sample["DataAnalysis"] = r(4, 5)

        elif role == "Data Scientist":
            sample["Python"] = r(4, 5)
            sample["DataAnalysis"] = r(4, 5)
            sample["HardSolved"] = r(15, 40)

        elif role == "Cloud Engineer":
            sample["Cloud"] = r(4, 5)
            sample["Networking"] = r(4, 5)

        elif role == "DevOps Engineer":
            sample["DevOps"] = r(4, 5)
            sample["Cloud"] = r(3, 5)

        elif role == "Mobile Developer":
            sample["Java"] = r(4, 5)

        elif role == "Cybersecurity Analyst":
            sample["Networking"] = r(4, 5)

        # Small random noise (10%)
        if random.random() < 0.1:
            sample["Python"] = r(2, 5)
            sample["DSA"] = r(2, 5)

        sample["Role"] = role
        data.append(sample)

    return data


def generate_dataset():
    all_data = []
    for role in ROLES:
        all_data.extend(generate_samples(role, 100))

    df = pd.DataFrame(all_data)
    df.to_csv("career_dataset.csv", index=False)
    print("Balanced dataset generated successfully.")


if __name__ == "__main__":
    generate_dataset()
