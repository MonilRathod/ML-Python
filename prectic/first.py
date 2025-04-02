class Person:
    def __init__(self, name, age, is_student):
        self.name = name
        self.age = age
        self.is_student = is_student


p1 = Person
p1.name = input("Enter your name: ")
p1.age = input("Enter your age: ")
is_student_input = input("Are you a student? (yes/no): ").strip().lower()
if is_student_input == "yes":
    p1.is_student = True
elif is_student_input == "no":
    p1.is_student = False
print(p1.name)
print(p1.age)
if p1.is_student:
    print(p1.name, "is a student.")
else:
    print(p1.name, "is not a student.")
