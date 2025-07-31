import pandas as pd
from faker import Faker
from copy import deepcopy
from random import choice

fake = Faker('en_IN')

# Controlled pools
DEGREE_POOL = ["B.Tech", "M.Tech", "MBA", "MD", "PhD", "B.Sc", "M.Sc"]
INDUSTRIES   = ["Technology", "Healthcare", "Finance", "Manufacturing", "Retail"]
RELS         = ["Spouse", "Child", "Parent", "Sibling", "Cousin"]

def anonymize_business_information(bi: dict) -> dict:
    bi = deepcopy(bi)
    bi['name']['value']    = fake.company()
    bi['name']['source']   = fake.word()
    bi['type']['value']    = fake.word()
    bi['type']['source']   = fake.word()
    addr = bi['address']
    addr['city']           = fake.city()
    addr['state']          = fake.state()
    addr['zip_code']       = fake.postcode()
    addr['country']        = fake.country()
    addr['source']         = fake.word()
    bi['business_address'] = fake.address().replace("\n", ", ")
    bi['industry']['value']  = choice(INDUSTRIES)
    bi['industry']['source'] = fake.word()
    bi['employees'] = [
        {'value': str(fake.random_int(1, 5000)), 'source': fake.word()}
        for _ in bi.get('employees', [])
    ]
    bi['business_role']['value']  = fake.job()
    bi['business_role']['source'] = fake.word()
    bi['annual_revenue']['value']  = f"${fake.random_int(1,50)}M"
    bi['annual_revenue']['source'] = fake.word()
    bi['years_in_business']['value']  = str(fake.random_int(1, 60))
    bi['years_in_business']['source'] = fake.word()
    return bi

def anonymize_career(career: dict) -> dict:
    career = deepcopy(career)
    history = []
    for entry in career.get('career_history', []):
        e = deepcopy(entry)
        e['type']          = fake.job()
        e['company']       = fake.company()
        e['start_date']    = fake.date(pattern="%Y-%m-%d")
        e['end_date']      = fake.date(pattern="%Y-%m-%d")
        e['position_held'] = fake.job()
        history.append(e)
    career['career_history']       = history
    career['career_highlights']['value']  = fake.sentence()
    career['career_highlights']['source'] = fake.word()
    return career

def anonymize_charitable_activity(ca: dict) -> dict:
    ca = deepcopy(ca)
    ca['charitable_roles'] = [
        {'title': fake.job(), 'source': fake.word()}
        for _ in ca.get('charitable_roles', [])
    ]
    givings = []
    for g in ca.get('charitable_giving', []):
        ng = {
            'type': fake.word(),
            'year': fake.date(pattern="%Y"),
            'donation': f"${fake.random_int(1,5)}00K",
            'organization': fake.company(),
            'source': fake.word()
        }
        givings.append(ng)
    ca['charitable_giving']           = givings
    ca['philanthropy_profile']['value']  = fake.sentence()
    ca['philanthropy_profile']['source'] = fake.word()
    for k, v in ca.get('charitable_giving_checklist', {}).items():
        if isinstance(v, bool):
            ca['charitable_giving_checklist'][k] = fake.boolean()
    return ca

def anonymize_connections(conns: list) -> list:
    out = []
    for c in conns:
        nc = deepcopy(c)
        nc['name']              = fake.name()
        nc['group']             = fake.word()
        nc['relationship_type'] = fake.word()
        nc['source']            = fake.word()
        out.append(nc)
    return out

def anonymize_education(edus: list) -> list:
    out = []
    for e in edus:
        ne = deepcopy(e)
        ne['institution']   = fake.company() + " University"
        ne['qualification'] = choice(DEGREE_POOL)
        ne['start_date']    = fake.date(pattern="%Y-%m-%d")
        ne['end_date']      = fake.date(pattern="%Y-%m-%d")
        ne['source']        = fake.word()
        out.append(ne)
    return out

def anonymize_family_details(fd: dict) -> dict:
    fd = deepcopy(fd)
    if 'members' in fd:
        members = []
        for m in fd['members']:
            nm = deepcopy(m)
            nm['name']         = fake.name()
            nm['age']          = str(fake.random_int(0, 100))
            nm['relationship'] = choice(RELS)
            nm['source']       = fake.word()
            members.append(nm)
        fd['members'] = members
    return fd

def anonymize_family_office(fo: dict) -> dict:
    fo = deepcopy(fo)
    fo['source'] = fake.word()
    new_members = []
    for member in fo.get('value', []):
        m = deepcopy(member)
        m['name']         = fake.name()
        m['relationship'] = choice(RELS)
        m['age']          = str(fake.random_int(0, 100))
        m['source']       = fake.word()
        new_members.append(m)
    fo['value'] = new_members
    return fo

def anonymize_private_foundation(pf: dict) -> dict:
    pf = deepcopy(pf)
    pf['source'] = fake.word()
    new_list = []
    for entry in pf.get('value', []):
        e = deepcopy(entry)
        e['total_assets'] = f"${fake.random_int(1,50)}M"
        e['total_income'] = f"${fake.random_int(1,10)}M"
        e['year']         = int(fake.year())
        e['phone']        = fake.phone_number()
        e['address']      = fake.address().replace("\n", ", ")
        e['source']       = fake.word()
        new_list.append(e)
    pf['value'] = new_list
    return pf

def anonymize_interests(ints: dict) -> dict:
    ints = deepcopy(ints)
    ints['interests']         = [
        {'value': fake.word(), 'source': fake.word()}
        for _ in ints.get('interests', [])
    ]
    ints['interests_remarks'] = {'value': fake.sentence(), 'source': fake.word()}
    return ints

def anonymize_personal_details(pd: dict) -> dict:
    pd = deepcopy(pd)
    pd['age']['value']            = str(fake.random_int(18, 80))
    pd['age']['source']           = fake.word()
    pd['bio']['value']            = fake.sentence()
    pd['bio']['source']           = fake.word()
    pd['location']['value']       = fake.address().replace("\n", ", ")
    pd['location']['source']      = fake.word()
    pd['marital_status']['value'] = fake.random_element(['Single','Married','Divorced'])
    pd['marital_status']['source']= fake.word()
    return pd

def anonymize_net_worth(nw: dict) -> dict:
    nw = deepcopy(nw)
    for section, lst in nw.items():
        if isinstance(lst, list):
            new_list = []
            for entry in lst:
                e = deepcopy(entry)
                e['value']  = f"${fake.random_int(1,20)}M"
                e['source'] = fake.word()
                new_list.append(e)
            nw[section] = new_list
    return nw

def anonymize_real_estate(re: list) -> list:
    out = []
    for p in re:
        np = {
            'taxes': f"${fake.random_int(1,5)}K",
            'address': fake.address().replace("\n", ", "),
            'property_type': fake.word(),
            'mortgage_value': f"${fake.random_int(50,300)}K",
            'mortgage_lender': fake.company(),
            'est_market_value': f"${fake.random_int(200,1000)}K",
            'source': fake.word()
        }
        out.append(np)
    return out

def anonymize_row(row: pd.Series) -> pd.Series:
    row = row.copy()
    # top‐level
    row['id']        = fake.random_number(digits=10, fix_len=True)
    row['name']      = fake.name()
    row['fa_id']     = fake.random_number(digits=6)
    row['added_on']  = fake.date(pattern="%Y-%m-%d")
    row['fa_name']   = fake.name()
    # nested
    row['business_information'] = anonymize_business_information(row['business_information'])
    row['career']               = anonymize_career(row['career'])
    row['charitable_activity']  = anonymize_charitable_activity(row['charitable_activity'])
    row['connections']          = anonymize_connections(row['connections'])
    row['education']            = anonymize_education(row['education'])
    row['family_details']       = anonymize_family_details(row['family_details'])
    row['family_office']        = anonymize_family_office(row['family_office'])
    row['private_foundation']   = anonymize_private_foundation(row['private_foundation'])
    row['interests']            = anonymize_interests(row['interests'])
    row['personal_details']     = anonymize_personal_details(row['personal_details'])
    row['net_worth']            = anonymize_net_worth(row['net_worth'])
    row['real_estate']          = anonymize_real_estate(row['real_estate'])
    # optional randomization
    row['completeness_score']   = fake.pyfloat(0,1)
    row['is_favourite']         = fake.boolean()
    return row

# Usage (do not run here):
# df = pd.read_pickle('your_df.pkl')
# anonymized_df = df.apply(anonymize_row, axis=1)