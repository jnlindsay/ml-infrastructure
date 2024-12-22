import hashlib

class Hash:
    @staticmethod
    def hash_to_string(obj):
        obj_bytes = str(obj).encode("utf-8")
        hash_obj = hashlib.sha256(obj_bytes)
        return hash_obj.hexdigest()