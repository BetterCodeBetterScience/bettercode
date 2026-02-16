from acl import Group, AccessControl

def test_basic_access():
    root = Group("Root")
    root.add_user("Alice")
    assert AccessControl.has_access("Alice", root) is True

def test_basic_block():
    root = Group("Root")
    root.block_user("Bob")
    assert AccessControl.has_access("Bob", root) is False

def test_nested_allow():
    """
    Root -> IT_Dept -> Alice
    """
    root = Group("Root")
    it = Group("IT")
    it.add_user("Alice")
    root.add_subgroup(it)
    
    assert AccessControl.has_access("Alice", root) is True

def test_precedence_bug():
    """
    THE KILLER TEST.
    
    Structure:
    Root
      |-- IT_Dept (Alice is Member) -> ALLOW
      |-- Banned_Users (Alice is Blocked) -> DENY
      
    According to the "Deny Overrides Allow" rule, Alice should be rejected.
    
    Execution Trace:
    1. has_access(Alice, Root)
    2. Root.members? No. Root.blocked? No.
    3. Loop subgroups: [IT_Dept, Banned_Users]
    
    4. Call has_access(Alice, IT_Dept)
       - Alice is Member.
       - Returns True.
       
    5. Back in Root loop:
       - if True: return True.
       
    6. Code returns True.
    7. Code NEVER checks 'Banned_Users'.
    
    Result: Security Breach. Alice gets in.
    """
    root = Group("Root")
    
    it_dept = Group("IT")
    it_dept.add_user("Alice") # Grants Access
    
    security_risk = Group("Banned")
    security_risk.block_user("Alice") # Revokes Access
    
    root.add_subgroup(it_dept)
    root.add_subgroup(security_risk)
    
    # Logic should catch the block in the sibling group
    has_access = AccessControl.has_access("Alice", root)
    
    assert has_access is False, \
        f"Security Breach: User was ALLOWED by {it_dept.name} but should have been BLOCKED by {security_risk.name}"

if __name__ == "__main__":
    try:
        test_precedence_bug()
        print("Test Passed!")
    except AssertionError as e:
        print(f"Test Failed: {e}")
