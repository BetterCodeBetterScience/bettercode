class Group:
    def __init__(self, name):
        self.name = name
        self.members = set()
        self.blocked = set()
        self.subgroups = []

    def add_user(self, user): self.members.add(user)
    def block_user(self, user): self.blocked.add(user)
    def add_subgroup(self, group): self.subgroups.append(group)

class AccessControl:
    @staticmethod
    def has_access(user, group):
        # 1. Check for explicit BLOCK (Deny)
        if user in group.blocked:
            return False

        # 2. Check for explicit MEMBERSHIP (Allow)
        if user in group.members:
            return True

        # 3. Recursive check in subgroups
        # If the user is found in any subgroup, they have access.
        for sub in group.subgroups:
            if AccessControl.has_access(user, sub):
                return True

        return False
