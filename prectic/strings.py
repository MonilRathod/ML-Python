course = '''Python's "awasome"'''
print(course)
print(course.upper())
print(course.replace("Python", "monil"))
print("python" in course)
print("Python" in course)
print(course.title())

while True:
    try:
        age = int(input("Enter your age: "))
        if age > 0:
            break
        else:
            print("Age must be a positive number.")
    except ValueError:
        print("Please enter a valid number.")
while True:
    try:
        b = str(input("Enter your gender: ")).strip().lower()
        if b == "male":
            break
        else:
            print("are you dumb")

    except ValueError:
        print("Please enter a valid string.")

if age >= 18 and b == "male":
    print("You are a man.")
else:
    print("whats up boi")
