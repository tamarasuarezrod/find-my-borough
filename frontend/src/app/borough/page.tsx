type Props = {
    params: {
      name: string;
    };
  };
  
  export default function BoroughPage({ params }: Props) {
    const boroughName = params.name.replaceAll('-', ' '); // opcional, para mostrarlo más bonito
  
    return (
      <div>
        <h1>Borough: {boroughName}</h1>
        <p>Info about {boroughName} coming soon...</p>
      </div>
    );
  }
  