from prov_recovery_model import ProvRecoveryModel

path = "data/xml"
test_doc = "data/xml/1510/10.1.1.1.1510v3.xml"
model = ProvRecoveryModel(path)
model.train()
model.generate_lineage(test_doc)
