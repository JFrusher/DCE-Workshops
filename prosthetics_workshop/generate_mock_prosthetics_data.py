from pathlib import Path

import pandas as pd


OUTPUT_FILE = Path('messy_prosthetics_data.csv')


PATIENT_ROWS = [
    {
        'Patient_ID': 'PT-001',
        'Full_Name': 'John Doe',
        'Age_Input': 28,
        'Implant_Date': '12/05/2024',
        'Amputation_Cause': 'Trauma ',
        'Device_Model_ID': 'M302',
        'Clinical_Notes': (
            'Patient was referred from the northern field clinic after a road traffic accident. '
            'The residual limb is healing well, but gait training is still required before discharge.'
        ),
    },
    {
        'Patient_ID': 'PT-002',
        'Full_Name': 'Amara Okafor',
        'Age_Input': '34',
        'Implant_Date': '2024-06-15',
        'Amputation_Cause': 'diabetes',
        'Device_Model_ID': 'M404',
        'Clinical_Notes': (
            'Patient arrived with a family member and requested follow-up support for socket comfort. '
            'Notes mention intermittent swelling and a need for diabetic foot education.'
        ),
    },
    {
        'Patient_ID': 'PT-003',
        'Full_Name': 'Miguel Santos',
        'Age_Input': 1995,
        'Implant_Date': 'May 12th 2024',
        'Amputation_Cause': 'TRAUMA',
        'Device_Model_ID': 'M302',
        'Clinical_Notes': (
            'The patient reported good progress after prosthetic fitting but needs a revised home exercise plan. '
            'Clinic staff noted that remote monitoring will help confirm long-term device comfort.'
        ),
    },
    {
        'Patient_ID': 'PT-004',
        'Full_Name': 'Priya Nair',
        'Age_Input': 'Don\'t know, maybe 40',
        'Implant_Date': '07/11/2023',
        'Amputation_Cause': 'blast injury',
        'Device_Model_ID': 'M505',
        'Clinical_Notes': (
            'This patient was seen during a mobile outreach visit after a conflict-related injury. '
            'The prosthetic review highlighted mild pressure marks and a need for adjustment.'
        ),
    },
    {
        'Patient_ID': 'PT-005',
        'Full_Name': 'Fatima Hassan',
        'Age_Input': 1988,
        'Implant_Date': '2023/09/01',
        'Amputation_Cause': 'Trauma',
        'Device_Model_ID': 'M404',
        'Clinical_Notes': (
            'The fitting session was completed successfully, and the patient was trained on cleaning routines. '
            'A review appointment was scheduled to monitor skin irritation and prosthesis alignment.'
        ),
    },
    {
        'Patient_ID': 'PT-006',
        'Full_Name': 'Daniel Brooks',
        'Age_Input': 'around 50',
        'Implant_Date': 'March 3 2024',
        'Amputation_Cause': 'diabetes ',
        'Device_Model_ID': 'M210',
        'Clinical_Notes': (
            'Patient reported improved balance after the last clinic session. '
            'The notes recommend continued physiotherapy and a check on the strap tension.'
        ),
    },
    {
        'Patient_ID': 'PT-007',
        'Full_Name': 'Lina Kim',
        'Age_Input': 19,
        'Implant_Date': '2024-01-20',
        'Amputation_Cause': ' TRAUMA',
        'Device_Model_ID': 'M302',
        'Clinical_Notes': (
            'The patient is recovering quickly and is learning how to use the device on uneven terrain. '
            'Staff recorded no major complications, only minor discomfort after long walking sessions.'
        ),
    },
    {
        'Patient_ID': 'PT-008',
        'Full_Name': 'Hassan Ali',
        'Age_Input': '41',
        'Implant_Date': '20/02/2024',
        'Amputation_Cause': 'blast Injury',
        'Device_Model_ID': 'M505',
        'Clinical_Notes': (
            'The field team flagged this record for a follow-up because the prosthetic sleeve loosened over time. '
            'Reassessment should confirm whether the current model still meets the patient\'s mobility needs.'
        ),
    },
    {
        'Patient_ID': 'PT-009',
        'Full_Name': 'Grace Mensah',
        'Age_Input': 52,
        'Implant_Date': 'April 5th 2024',
        'Amputation_Cause': 'trauma',
        'Device_Model_ID': 'M404',
        'Clinical_Notes': (
            'Patient was calm and cooperative during the fitting session and asked detailed questions about maintenance. '
            'The clinician documented a successful first trial with a strong initial fit.'
        ),
    },
    {
        'Patient_ID': 'PT-010',
        'Full_Name': 'Oliver Bennett',
        'Age_Input': 1990,
        'Implant_Date': '2024-08-30',
        'Amputation_Cause': ' DIABETES',
        'Device_Model_ID': 'M210',
        'Clinical_Notes': (
            'The patient completed the standard intake form but left several fields blank, requiring manual review. '
            'Clinic notes mention good progress with physiotherapy and strong family support.'
        ),
    },
    {
        'Patient_ID': 'PT-011',
        'Full_Name': 'Nia Williams',
        'Age_Input': '45',
        'Implant_Date': '11/12/2023',
        'Amputation_Cause': 'Trauma ',
        'Device_Model_ID': 'M302',
        'Clinical_Notes': (
            'Patient requested additional training materials because she will be travelling to a remote village. '
            'The clinic team recorded no adverse events, but the socket may need padding after travel.'
        ),
    },
    {
        'Patient_ID': 'PT-012',
        'Full_Name': 'Kwame Boateng',
        'Age_Input': 'approx 36',
        'Implant_Date': '2023-10-14',
        'Amputation_Cause': 'diabetes',
        'Device_Model_ID': 'M505',
        'Clinical_Notes': (
            'The record indicates the patient is using the prosthetic device daily and tolerating it well. '
            'Additional observations focus on moisture management and safe cleaning procedures.'
        ),
    },
    {
        'Patient_ID': 'PT-013',
        'Full_Name': 'Sofia Martinez',
        'Age_Input': 2001,
        'Implant_Date': 'June 9th 2024',
        'Amputation_Cause': 'TRAUMA ',
        'Device_Model_ID': 'M404',
        'Clinical_Notes': (
            'Patient was seen after discharge from the regional hospital and now requires ongoing community follow-up. '
            'The notes recommend a device review in two weeks to ensure the alignment remains stable.'
        ),
    },
    {
        'Patient_ID': 'PT-014',
        'Full_Name': 'Amina Yusuf',
        'Age_Input': '30',
        'Implant_Date': '2024/04/18',
        'Amputation_Cause': 'blast injury ',
        'Device_Model_ID': 'M210',
        'Clinical_Notes': (
            'This patient completed the remote intake process with limited internet connectivity. '
            'The clinical note mentions a successful first fit, though gait confidence is still developing.'
        ),
    },
    {
        'Patient_ID': 'PT-015',
        'Full_Name': 'Ethan Clarke',
        'Age_Input': 'maybe 27',
        'Implant_Date': '15/07/2024',
        'Amputation_Cause': 'trauma',
        'Device_Model_ID': 'M302',
        'Clinical_Notes': (
            'Patient reported that the prosthesis helped them return to basic daily activities after injury. '
            'The follow-up plan includes continued strength work and a review of comfort after prolonged wear.'
        ),
    },
]


DEVICE_INVENTORY = {
    'M210': {
        'Model_Name': 'AeroStep Lite',
        'Cost_USD': 950,
        'Manufacturer': 'Stride MedTech',
    },
    'M302': {
        'Model_Name': 'FlexFoot Alpha',
        'Cost_USD': 1200,
        'Manufacturer': 'Global Limb Works',
    },
    'M404': {
        'Model_Name': 'CrestFit Pro',
        'Cost_USD': 1850,
        'Manufacturer': 'OrthoSphere',
    },
    'M505': {
        'Model_Name': 'AdaptMax Field',
        'Cost_USD': 2100,
        'Manufacturer': 'Humanity Mobility Systems',
    },
}


def build_patient_dataframe() -> pd.DataFrame:
    """Return the intentionally messy patient intake dataset."""
    df = pd.DataFrame(PATIENT_ROWS)
    expected_columns = [
        'Patient_ID',
        'Full_Name',
        'Age_Input',
        'Implant_Date',
        'Amputation_Cause',
        'Device_Model_ID',
        'Clinical_Notes',
    ]
    df = df[expected_columns]
    assert len(df) == 15, 'The workshop dataset must contain exactly 15 rows.'
    return df


def build_device_inventory_dataframe() -> pd.DataFrame:
    """Return the small reference table used for the relational join exercise."""
    inventory_df = pd.DataFrame.from_dict(DEVICE_INVENTORY, orient='index').reset_index()
    inventory_df = inventory_df.rename(columns={'index': 'Device_Model_ID'})
    return inventory_df[['Device_Model_ID', 'Model_Name', 'Cost_USD', 'Manufacturer']]


def main() -> None:
    patient_df = build_patient_dataframe()
    patient_df.to_csv(OUTPUT_FILE, index=False)

    inventory_df = build_device_inventory_dataframe()

    print(f'Wrote {len(patient_df)} rows to {OUTPUT_FILE.resolve()}')
    print('\nDevice inventory reference table:')
    print(inventory_df.to_string(index=False))


if __name__ == '__main__':
    main()
