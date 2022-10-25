from FairEMOL.EvolutionaryCodes.data.objects.Adult import Adult
from FairEMOL.EvolutionaryCodes.data.objects.German import German
from FairEMOL.EvolutionaryCodes.data.objects.PropublicaRecidivism import PropublicaRecidivism
from FairEMOL.EvolutionaryCodes.data.objects.Bank import Bank
from FairEMOL.EvolutionaryCodes.data.objects.Default import Default
from FairEMOL.EvolutionaryCodes.data.objects.LSAT import LSAT
from FairEMOL.EvolutionaryCodes.data.objects.Dutch import Dutch
from FairEMOL.EvolutionaryCodes.data.objects.Student_mat import Student_mat
from FairEMOL.EvolutionaryCodes.data.objects.Drug_consumption import Drug_consumption
from FairEMOL.EvolutionaryCodes.data.objects.Heart_failure import Heart_failure
from FairEMOL.EvolutionaryCodes.data.objects.Student_academics import Student_academics
from FairEMOL.EvolutionaryCodes.data.objects.Diabetes import Diabetes
from FairEMOL.EvolutionaryCodes.data.objects.Student_performance import Student_performance
from FairEMOL.EvolutionaryCodes.data.objects.Patient_treatment import Patient_treatment
from FairEMOL.EvolutionaryCodes.data.objects.IBM_employee import IBM_employee

DATASETS = [
    Dutch(),
    Adult(),
    German(),
    PropublicaRecidivism(),
    Bank(),
    Default(),
    LSAT(),
    Drug_consumption(),
    Heart_failure(),
    Student_academics(),
    Diabetes(),
    Student_performance(),
    Patient_treatment(),
    IBM_employee(),
    Student_mat(),
]


def get_dataset_names():
    names = []
    for dataset in DATASETS:
        names.append(dataset.get_dataset_name())
    return names


def add_dataset(dataset):
    DATASETS.append(dataset)


def get_dataset_by_name(name):
    for ds in DATASETS:
        if ds.get_dataset_name() == name:
            return ds
    raise Exception("No dataset with name %s could be found." % name)
