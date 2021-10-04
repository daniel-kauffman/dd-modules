import unittest

import numconv


class TestNumberConversion(unittest.TestCase):
    
    def test_tens(self):
        text = "ten"
        self.assertEqual(numconv.parse_number(text.split()), "10")
        text = "twenty one"
        self.assertEqual(numconv.parse_number(text.split()), "21")
        text = "thirty two"
        self.assertEqual(numconv.parse_number(text.split()), "32")
        text = "forty three"
        self.assertEqual(numconv.parse_number(text.split()), "43")
        text = "fifty four"
        self.assertEqual(numconv.parse_number(text.split()), "54")
        text = "sixty five"
        self.assertEqual(numconv.parse_number(text.split()), "65")
        text = "seventy six"
        self.assertEqual(numconv.parse_number(text.split()), "76")
        text = "eighty seven"
        self.assertEqual(numconv.parse_number(text.split()), "87")
        text = "ninety eight"
        self.assertEqual(numconv.parse_number(text.split()), "98")
    
    def test_hundreds(self):
        text = "a hundred"
        self.assertEqual(numconv.parse_number(text.split()), "100")
        text = "one hundred"
        self.assertEqual(numconv.parse_number(text.split()), "100")
        text = "two hundred and fifty"
        self.assertEqual(numconv.parse_number(text.split()), "250")
    
    def test_thousands(self):
        text = "a thousand"
        self.assertEqual(numconv.parse_number(text.split()), "1,000")
        text = "one thousand"
        self.assertEqual(numconv.parse_number(text.split()), "1,000")
        text = "twenty five thousand"
        self.assertEqual(numconv.parse_number(text.split()), "25,000")
        text = "twenty five thousand two hundred fifty"
        self.assertEqual(numconv.parse_number(text.split()), "25,250")
        text = "one hundred twenty three thousand four hundred fifty six"
        self.assertEqual(numconv.parse_number(text.split()), "123,456")
    
    def test_others(self):
        text = "two million one hundred thousand"
        self.assertEqual(numconv.parse_number(text.split()), "2,100,000")
        text = "three billion twenty million one hundred thousand"
        self.assertEqual(numconv.parse_number(text.split()), "3,020,100,000")
        text = "one point two million"
        self.assertEqual(numconv.parse_number(text.split()), "1,200,000")
    
    def test_fractions(self):
        text = "oh point oh"
        self.assertEqual(numconv.parse_number(text.split()), "0.0")
        text = "oh point zero"
        self.assertEqual(numconv.parse_number(text.split()), "0.0")
        text = "zero point oh"
        self.assertEqual(numconv.parse_number(text.split()), "0.0")
        text = "zero point zero"
        self.assertEqual(numconv.parse_number(text.split()), "0.0")
        text = "oh point oh one"
        self.assertEqual(numconv.parse_number(text.split()), "0.01")
        text = "one point oh"
        self.assertEqual(numconv.parse_number(text.split()), "1.0")
        text = "one point zero"
        self.assertEqual(numconv.parse_number(text.split()), "1.0")
        text = "oh point two one"
        self.assertEqual(numconv.parse_number(text.split()), "0.21")
        text = "oh point twenty one"
        self.assertEqual(numconv.parse_number(text.split()), "0.21")
        text = "zero point one one"
        self.assertEqual(numconv.parse_number(text.split()), "0.11")
        text = "zero point eleven"
        self.assertEqual(numconv.parse_number(text.split()), "0.11")
        text = "two point one"
        self.assertEqual(numconv.parse_number(text.split()), "2.1")
        text = "three point two one"
        self.assertEqual(numconv.parse_number(text.split()), "3.21")
        text = "three point twenty one"
        self.assertEqual(numconv.parse_number(text.split()), "3.21")
        text = "one hundred twenty three point four"
        self.assertEqual(numconv.parse_number(text.split()), "123.4")
    
    def test_years(self):
        text = "seventeen seventy six"
        self.assertEqual(numconv.parse_number(text.split()), "1776")
        text = "nineteen oh six"
        self.assertEqual(numconv.parse_number(text.split()), "1906")
        text = "twenty sixteen"
        self.assertEqual(numconv.parse_number(text.split()), "2016")
        text = "two thousand twenty one"
        self.assertEqual(numconv.parse_number(text.split()), "2021")
    
    def test_bills(self):
        text = "one two"
        self.assertEqual(numconv.parse_number(text.split()), "12")
        text = "three four five"
        self.assertEqual(numconv.parse_number(text.split()), "345")
        text = "six seven eight nine"
        self.assertEqual(numconv.parse_number(text.split()), "6789")
        text = "one twenty three"
        self.assertEqual(numconv.parse_number(text.split()), "123")


if __name__ == "__main__":
    unittest.main()
