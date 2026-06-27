import logging

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def valid_subjects(records: list[dict], max_age: int = 120) -> list[str]:
    logger.info("Validating %d subject records", len(records))

    valid = []
    for record in records:
        subject_id = record["id"]
        age = record["age"]
        logger.debug("Checking %s (age=%d)", subject_id, age)

        if age > max_age:
            logger.warning("Implausible age %d for %s; skipping", age, subject_id)
            continue

        valid.append(subject_id)

    logger.info("Kept %d of %d records", len(valid), len(records))
    return valid


if __name__ == "__main__":
    records = [
        {"id": "S01", "age": 34},
        {"id": "S02", "age": 32757},
        {"id": "S03", "age": 28},
    ]
    valid_subjects(records)
