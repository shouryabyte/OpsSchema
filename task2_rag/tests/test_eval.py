from src.evaluate import evidence_hit
class X:
 def __init__(self,t):self.text=t
def test_exact_span(): assert evidence_hit([X('NASA systems engineering is iterative.')],'systems engineering is iterative')
