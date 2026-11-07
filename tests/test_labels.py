import pytest

from src.data.labels import folder_name, parse_class_name


@pytest.mark.parametrize("name, make, model, year, body", [
    ("AM General Hummer SUV 2000", "AM General", "Hummer SUV", 2000, "SUV"),
    ("Aston Martin V8 Vantage Coupe 2012", "Aston Martin", "V8 Vantage Coupe", 2012, "Coupe"),
    ("Land Rover Range Rover SUV 2012", "Land Rover", "Range Rover SUV", 2012, "SUV"),
    ("Mercedes-Benz C-Class Sedan 2012", "Mercedes-Benz", "C-Class Sedan", 2012, "Sedan"),
    ("Chevrolet Silverado 1500 Classic Extended Cab 2007", "Chevrolet", "Silverado 1500 Classic Extended Cab", 2007, "Pickup"),
    ("Ford Ranger SuperCab 2011", "Ford", "Ranger SuperCab", 2011, "Pickup"),
    ("Ram C/V Cargo Van Minivan 2012", "Ram", "C/V Cargo Van Minivan", 2012, "Minivan"),
    ("Chevrolet Express Cargo Van 2007", "Chevrolet", "Express Cargo Van", 2007, "Van"),
    ("Chevrolet Cobalt SS 2010", "Chevrolet", "Cobalt SS", 2010, "Other"),
    ("smart fortwo Convertible 2012", "smart", "fortwo Convertible", 2012, "Convertible"),
])
def test_parse(name, make, model, year, body):
    lab = parse_class_name(name)
    assert (lab.make, lab.model, lab.year, lab.body_type) == (make, model, year, body)


def test_bad_name():
    with pytest.raises(ValueError):
        parse_class_name("No Year Here")


def test_folder_name():
    assert folder_name("Ram C/V Cargo Van Minivan 2012") == "Ram C-V Cargo Van Minivan 2012"
