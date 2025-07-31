import pandas as pd
from faker import Faker
from copy import deepcopy
from random import choice

fake = Faker('en_IN')

# Controlled pools
DEGREE_POOL = ["B.Tech", "M.Tech", "MBA", "MD", "PhD", "B.Sc", "M.Sc"]
INDUSTRIES   = ["Technology", "Healthcare", "Finance", "Manufacturing", "Retail"]
RELS         = ["Spouse", "Child", "Parent", "Sibling", "Cousin"]

def replace_if(orig, new):
    """
    Return new if orig is non-empty/non-null.
    Otherwise leave orig as is.
    Treats '', None, [], {} as 'empty'.
    """
    if orig in (None, "", [], {}):
        return orig
    return new

# 1) Top-level name now a dict
def anonymize_name_field(n: dict) -> dict:
    n = deepcopy(n)
    n['value']  = replace_if(n.get('value'), fake.name())
    n['source'] = replace_if(n.get('source'), fake.word())
    return n

def anonymize_business_information(bi: dict) -> dict:
    bi = deepcopy(bi)
    # name & type sub-dicts
    bi['name']['value']    = replace_if(bi['name'].get('value'), fake.company())
    bi['name']['source']   = replace_if(bi['name'].get('source'), fake.word())
    bi['type']['value']    = replace_if(bi['type'].get('value'), fake.word())
    bi['type']['source']   = replace_if(bi['type'].get('source'), fake.word())
    # address sub-dict
    addr = bi['address']
    addr['city']     = replace_if(addr.get('city'), fake.city())
    addr['state']    = replace_if(addr.get('state'), fake.state())
    addr['country']  = replace_if(addr.get('country'), fake.country())
    addr['zip_code'] = replace_if(addr.get('zip_code'), fake.postcode())
    addr['source']   = replace_if(addr.get('source'), fake.word())
    # business_address
    bi['business_address'] = replace_if(
        bi.get('business_address'),
        fake.address().replace("\n", ", ")
    )
    # industry
    bi['industry']['value']  = replace_if(bi['industry'].get('value'), choice(INDUSTRIES))
    bi['industry']['source'] = replace_if(bi['industry'].get('source'), fake.word())
    # employees list
    if bi.get('employees'):
        new_emp = []
        for e in bi['employees']:
            orig_val = e.get('value')
            new_emp.append({
                'value':  replace_if(orig_val, str(fake.random_int(1,5000))),
                'source': replace_if(e.get('source'), fake.word())
            })
        bi['employees'] = new_emp
    # other flat fields
    bi['business_role']['value']  = replace_if(bi['business_role'].get('value'), fake.job())
    bi['business_role']['source'] = replace_if(bi['business_role'].get('source'), fake.word())
    bi['annual_revenue']['value']  = replace_if(
        bi['annual_revenue'].get('value'),
        f"${fake.random_int(1,50)}M"
    )
    bi['annual_revenue']['source'] = replace_if(bi['annual_revenue'].get('source'), fake.word())
    bi['years_in_business']['value']  = replace_if(
        bi['years_in_business'].get('value'),
        str(fake.random_int(1,60))
    )
    bi['years_in_business']['source'] = replace_if(bi['years_in_business'].get('source'), fake.word())
    return bi

def anonymize_career(career: dict) -> dict:
    career = deepcopy(career)
    if career.get('career_history'):
        new_hist = []
        for entry in career['career_history']:
            e = deepcopy(entry)
            e['type']          = replace_if(e.get('type'), fake.job())
            e['company']       = replace_if(e.get('company'), fake.company())
            e['start_date']    = replace_if(e.get('start_date'), fake.date(pattern="%Y-%m-%d"))
            e['end_date']      = replace_if(e.get('end_date'), fake.date(pattern="%Y-%m-%d"))
            e['position_held'] = replace_if(e.get('position_held'), fake.job())
            new_hist.append(e)
        career['career_history'] = new_hist
    career['career_highlights']['value']  = replace_if(
        career['career_highlights'].get('value'),
        fake.sentence()
    )
    career['career_highlights']['source'] = replace_if(
        career['career_highlights'].get('source'),
        fake.word()
    )
    return career

def anonymize_charitable_activity(ca: dict) -> dict:
    ca = deepcopy(ca)
    if ca.get('charitable_roles'):
        ca['charitable_roles'] = [
            {
                'title':  replace_if(r.get('title'), fake.job()),
                'source': replace_if(r.get('source'), fake.word())
            }
            for r in ca['charitable_roles']
        ]
    if ca.get('charitable_giving'):
        new_g = []
        for g in ca['charitable_giving']:
            ng = {
                'type':         replace_if(g.get('type'), fake.word()),
                'year':         replace_if(g.get('year'), fake.date(pattern="%Y")),
                'donation':     replace_if(g.get('donation'), f"${fake.random_int(1,5)}00K"),
                'organization': replace_if(g.get('organization'), fake.company()),
                'source':       replace_if(g.get('source'), fake.word())
            }
            new_g.append(ng)
        ca['charitable_giving'] = new_g
    ca['philanthropy_profile']['value']  = replace_if(
        ca['philanthropy_profile'].get('value'),
        fake.sentence()
    )
    ca['philanthropy_profile']['source'] = replace_if(
        ca['philanthropy_profile'].get('source'),
        fake.word()
    )
    # checklist: only booleans
    for k,v in ca.get('charitable_giving_checklist', {}).items():
        if isinstance(v, bool):
            ca['charitable_giving_checklist'][k] = replace_if(v, fake.boolean())
    return ca

def anonymize_connections(conns: list) -> list:
    out = []
    for c in conns:
        nc = deepcopy(c)
        nc['name']              = replace_if(nc.get('name'), fake.name())
        nc['group']             = replace_if(nc.get('group'), fake.word())
        nc['relationship_type'] = replace_if(nc.get('relationship_type'), fake.word())
        nc['source']            = replace_if(nc.get('source'), fake.word())
        out.append(nc)
    return out

def anonymize_education(edus: list) -> list:
    out = []
    for e in edus:
        ne = deepcopy(e)
        ne['institution']   = replace_if(ne.get('institution'), fake.company() + " University")
        ne['qualification'] = replace_if(ne.get('qualification'), choice(DEGREE_POOL))
        ne['start_date']    = replace_if(ne.get('start_date'), fake.date(pattern="%Y-%m-%d"))
        ne['end_date']      = replace_if(ne.get('end_date'), fake.date(pattern="%Y-%m-%d"))
        ne['source']        = replace_if(ne.get('source'), fake.word())
        out.append(ne)
    return out

def anonymize_family_details(fd: dict) -> dict:
    fd = deepcopy(fd)
    if fd.get('members'):
        members = []
        for m in fd['members']:
            nm = deepcopy(m)
            nm['name']         = replace_if(nm.get('name'), fake.name())
            nm['age']          = replace_if(nm.get('age'), str(fake.random_int(0,100)))
            nm['relationship'] = replace_if(nm.get('relationship'), choice(RELS))
            nm['source']       = replace_if(nm.get('source'), fake.word())
            members.append(nm)
        fd['members'] = members
    return fd

def anonymize_family_office(fo: dict) -> dict:
    fo = deepcopy(fo)
    fo['source'] = replace_if(fo.get('source'), fake.word())
    if fo.get('value'):
        new_val = []
        for m in fo['value']:
            nm = deepcopy(m)
            nm['name']         = replace_if(nm.get('name'), fake.name())
            nm['relationship'] = replace_if(nm.get('relationship'), choice(RELS))
            nm['age']          = replace_if(nm.get('age'), str(fake.random_int(0,100)))
            nm['source']       = replace_if(nm.get('source'), fake.word())
            new_val.append(nm)
        fo['value'] = new_val
    return fo

def anonymize_private_foundation(pf: dict) -> dict:
    pf = deepcopy(pf)
    pf['source'] = replace_if(pf.get('source'), fake.word())
    if pf.get('value'):
        new_list = []
        for e in pf['value']:
            ne = deepcopy(e)
            ne['total_assets'] = replace_if(ne.get('total_assets'), f"${fake.random_int(1,50)}M")
            ne['total_income'] = replace_if(ne.get('total_income'), f"${fake.random_int(1,10)}M")
            ne['year']         = replace_if(ne.get('year'), int(fake.year()))
            ne['phone']        = replace_if(ne.get('phone'), fake.phone_number())
            ne['address']      = replace_if(ne.get('address'), fake.address().replace("\n",", "))
            ne['source']       = replace_if(ne.get('source'), fake.word())
            new_list.append(ne)
        pf['value'] = new_list
    return pf

def anonymize_interests(ints: dict) -> dict:
    ints = deepcopy(ints)
    if ints.get('interests'):
        ints['interests'] = [
            {
                'value':  replace_if(i.get('value'), fake.word()),
                'source': replace_if(i.get('source'), fake.word())
            }
            for i in ints['interests']
        ]
    ints['interests_remarks']['value']  = replace_if(
        ints['interests_remarks'].get('value'),
        fake.sentence()
    )
    ints['interests_remarks']['source'] = replace_if(
        ints['interests_remarks'].get('source'),
        fake.word()
    )
    return ints

def anonymize_personal_details(pd: dict) -> dict:
    pd = deepcopy(pd)
    pd['age']['value']            = replace_if(pd['age'].get('value'), str(fake.random_int(18,80)))
    pd['age']['source']           = replace_if(pd['age'].get('source'), fake.word())
    pd['bio']['value']            = replace_if(pd['bio'].get('value'), fake.sentence())
    pd['bio']['source']           = replace_if(pd['bio'].get('source'), fake.word())
    pd['location']['value']       = replace_if(pd['location'].get('value'), fake.address().replace("\n",", "))
    pd['location']['source']      = replace_if(pd['location'].get('source'), fake.word())
    pd['marital_status']['value'] = replace_if(
        pd['marital_status'].get('value'),
        fake.random_element(['Single','Married','Divorced'])
    )
    pd['marital_status']['source']= replace_if(pd['marital_status'].get('source'), fake.word())
    return pd

def anonymize_net_worth(nw: dict) -> dict:
    nw = deepcopy(nw)
    for section, lst in nw.items():
        if isinstance(lst, list):
            new_list = []
            for e in lst:
                ne = deepcopy(e)
                ne['value']  = replace_if(ne.get('value'), f"${fake.random_int(1,20)}M")
                ne['source'] = replace_if(ne.get('source'), fake.word())
                new_list.append(ne)
            nw[section] = new_list
    return nw

def anonymize_real_estate(re: list) -> list:
    out = []
    for p in re:
        np = deepcopy(p)
        np['taxes']           = replace_if(np.get('taxes'), f"${fake.random_int(1,5)}K")
        np['address']         = replace_if(np.get('address'), fake.address().replace("\n",", "))
        np['property_type']   = replace_if(np.get('property_type'), fake.word())
        np['mortgage_value']  = replace_if(np.get('mortgage_value'), f"${fake.random_int(50,300)}K")
        np['mortgage_lender'] = replace_if(np.get('mortgage_lender'), fake.company())
        np['est_market_value']= replace_if(np.get('est_market_value'), f"${fake.random_int(200,1000)}K")
        np['source']          = replace_if(np.get('source'), fake.word())
        out.append(np)
    return out

def anonymize_row(row: pd.Series) -> pd.Series:
    row = row.copy()
    # — name as dict —
    row['name'] = anonymize_name_field({'value': row.get('name'), 'source': row.get('name_source','')})
    # optionally name_source could be a separate column; else we overwrite source
    
    # — other top‐level scalars —
    row['id']        = replace_if(row.get('id'), fake.random_number(digits=10, fix_len=True))
    row['fa_id']     = replace_if(row.get('fa_id'), fake.random_number(digits=6))
    row['added_on']  = replace_if(row.get('added_on'), fake.date(pattern="%Y-%m-%d'))
    row['fa_name']   = replace_if(row.get('fa_name'), fake.name())
    
    # — nested —
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
    
    # — optional booleans/scores —
    row['completeness_score']   = replace_if(row.get('completeness_score'), fake.pyfloat(0,1))
    row['is_favourite']         = replace_if(row.get('is_favourite'), fake.boolean())
    return row

# Usage (do not run here):
# df = pd.read_pickle('your_df.pkl')
# anonymized_df = df.apply(anonymize_row, axis=1)